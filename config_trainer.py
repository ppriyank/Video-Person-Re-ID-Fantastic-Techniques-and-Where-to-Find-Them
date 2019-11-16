from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
# from tools import * 
import models

from loss import CrossEntropyLabelSmooth, TripletLoss , CenterLoss , OSM_CAA_Loss
from tools.transforms2 import *
from tools.scheduler import WarmupMultiStepLR
from tools.utils import AverageMeter, Logger, save_checkpoint , resume_from_checkpoint
from tools.eval_metrics import evaluate
from tools.samplers import RandomIdentitySampler
from tools.video_loader import VideoDataset 
import tools.data_manager as data_manager

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=400, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--use-OSMCAA', action='store_true', default=False,
                    help="Use OSM CAA loss in addition to triplet")
parser.add_argument('--cl-centers', action='store_true', default=False,
                    help="Use cl centers verison of OSM CAA loss")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=40, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='/home/jiyang/Workspace/Works/video-person-reid/3dconv-person-reid/pretrained_models/resnet-50-kinetics.pth', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--epochs-eval', default=[10 * i for i in range(6,80)], type=list)
parser.add_argument('--name', '--model_name', type=str, default='_bot_')
parser.add_argument('--validation-training', action='store_true', help="more useful for validation")
parser.add_argument('--resume-training', action='store_true', help="Continue training")
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-f', '--focus', type=str, default='map', help="map,rerank_map")
parser.add_argument('-opt', '--opt', type=str, default='3', help="choose opt")

args = parser.parse_args()

torch.manual_seed(args.seed)

# args.gpu_devices = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()

cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)

dataset = data_manager.init_dataset(name=args.dataset)
pin_memory = True if use_gpu else False

transform_train = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            Random2DTranslation(args.height, args.width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing2(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])

transform_test = transforms.Compose([
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




import configparser
config = configparser.ConfigParser()

dirpath = os.getcwd() 
opt = args.opt
    

if args.dataset == "mars":
    print("USING MARS CONFIG")
    print("cl_centers.conf" , "========== ", opt ,"===============")
    config.read(dirpath + "/tools/cl_centers.conf")        
    sigma = float(config[opt]['sigma'])
    alpha =  float(config[opt]['alpha'])
    l = float(config[opt]['l'])
    margin =  float(config[opt]['margin'])
    beta_ratio = float(config[opt]['beta_ratio'])
    gamma  = float(config[opt]['gamma'])
    weight_decay = float(config[opt]['weight_decay'])
    if 'batch_size' in config[opt]:
        batch_size = int(config[opt]['batch_size'])
    else:
        batch_size = 32


elif args.dataset == "prid":
    print("USING PRID CONFIG")
    print("prid.conf" , "========== ", opt ,"===============")
    config.read(dirpath + "/tools/prid.conf")        
    sigma = float(config[opt]['sigma'])
    alpha =  float(config[opt]['alpha'])
    l = float(config[opt]['l'])
    margin =  float(config[opt]['margin'])
    beta_ratio = float(config[opt]['beta_ratio'])
    gamma  = float(config[opt]['gamma'])
    weight_decay = float(config[opt]['weight_decay'])
    if 'batch_size' in config[opt]:
        batch_size = int(config[opt]['batch_size'])
    else:
        batch_size = 32




trainloader = DataLoader(
    VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
    sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
    batch_size=args.train_batch, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=True,
)

queryloader = DataLoader(
    VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)


galleryloader = DataLoader(
    VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)



args.arch = "ResNet50ta_bt"
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

criterion_htri = TripletLoss(margin, 'cosine')
criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids)
criterion_center_loss = CenterLoss(use_gpu=use_gpu)

if args.use_OSMCAA:
    print("USING OSM LOSS")
    print ("config, alpha = %f  sigma = %f  l=%f"%(alpha, sigma, l )  )
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma )
else:
    criterion_osm_caa = None


if args.cl_centers:
    print("USING CL CENTERS")
    print ("config, alpha = %f  sigma = %f  l=%f"%(alpha, sigma, l )  )
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma )


base_learning_rate =  0.00035
params = []
for key, value in model.named_parameters():
    if not value.requires_grad:
        continue
    lr = base_learning_rate
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


optimizer = torch.optim.Adam(params)
scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)
optimizer_center = torch.optim.SGD(criterion_center_loss.parameters(), lr=0.5)

start_epoch = args.start_epoch
if use_gpu:
    model = nn.DataParallel(model).cuda()


best_rank1 = -np.inf


def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa=None):
    model.train()
    losses = AverageMeter()
    cetner_loss_weight = 0.0005
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        # print(batch_idx)
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
            # 32,4,3,224,112 , 32
        imgs, pids = Variable(imgs), Variable(pids)
        outputs, features = model(imgs)
        ide_loss = criterion_xent(outputs , pids)
        triplet_loss = criterion_htri(features, features, features, pids, pids, pids)
        center_loss = criterion_center_loss(features, pids)
        if args.use_OSMCAA:
            if use_gpu:
                osm_caa_loss = criterion_osm_caa(features, pids , model.module.classifier.classifier.weight.t())         
            else:
                osm_caa_loss = criterion_osm_caa(features, pids , model.classifier.classifier.weight.t()) 
            loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss
        elif args.cl_centers : 
            osm_caa_loss = criterion_osm_caa(features, pids , criterion_center_loss.centers.t()) 
            loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss
        else:
            loss = ide_loss + triplet_loss  + center_loss * cetner_loss_weight
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in criterion_center_loss.parameters():
            param.grad.data *= (1./cetner_loss_weight)
        optimizer_center.step()
        losses.update(loss.data.item(), pids.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            if args.use_OSMCAA or args.cl_centers:
                print("Batch {}/{}\t TripletLoss  ({:.6f}) OSM Loss: ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(batch_idx+1, len(trainloader) ,triplet_loss.item(), osm_caa_loss.item() ,losses.val, losses.avg))        
            else:
                print("Batch {}/{}\t TripletLoss  ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(batch_idx+1, len(trainloader) ,triplet_loss.item(),losses.val, losses.avg))        
        
        
        


from tools.eval_metrics import evaluate , re_ranking
def test_rerank(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        assert(b==1)
        imgs = imgs.view(b*n, s, c, h, w)
        features = model(imgs)
        features = features.view(n, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s , c, h, w)
        assert(b==1)
        features = model(imgs)
        features = features.view(n, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    distmat_rerank = re_ranking(qf,gf)
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Rerank Computing CMC and mAP")
    re_rank_cmc, re_rank_mAP = evaluate(distmat_rerank, q_pids, g_pids, q_camids, g_camids)
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    if 'mars' in args.dataset :
        print("mAP: {:.1%} vs {:.1%}".format(mAP, re_rank_mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%} vs {:.1%}".format(r, cmc[r-1], re_rank_cmc[r-1]))
    print("------------------")
    if 'mars' not in args.dataset :
        print("Dataset not MARS : instead", args.dataset)
        return cmc[0]
    else:
        if args.focus == "map":
            print("returning map")
            return mAP
        else:
            print("returning re-rank")
            return re_rank_mAP







args.save_dir = "/scratch/pp1953/resnet/"
args.arch += args.name +"_" + str(args.opt) + "_"
is_best = 0
prev_best = 0 

if type(args.epochs_eval[0]) != int :
    args.epochs_eval = map(int,  "".join([vals for vals in args.epochs_eval if vals != '[' and vals != ']']).split(",") )

print (args.arch)




if not args.validation_training:
    
    print(args.epochs_eval)
    if not args.evaluate :
        for epoch in range(0, args.max_epoch):
            print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
            train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa )
            if args.stepsize > 0 : scheduler.step()
            if epoch in args.epochs_eval:
                re_rank_mAP = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
                if re_rank_mAP > prev_best:
                    prev_best = re_rank_mAP
                    if use_gpu:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    save_checkpoint({
                        'state_dict': state_dict,
                        }, is_best, osp.join(args.save_dir, args.arch + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))                

    else:
        for epoch in args.epochs_eval:
            checkpoint = torch.load(osp.join(args.save_dir, args.arch + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))
            state_dict = {}
            for key in checkpoint['state_dict']:
                    state_dict["module." + key] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict,  strict=True)
            rank1 = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)

else:
    print("evaluation at every 10 epochs, Highly GPU/CPU expensive process, avoid running anything in Parallel")
    factor = 10 
    print(args.epochs_eval)
    print("====")
    print(args.start_epoch)
    for epoch in range(args.start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa )
        if args.stepsize > 0 : scheduler.step()
        if epoch in args.epochs_eval:
            re_rank_mAP = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
            if re_rank_mAP > prev_best:
                prev_best = re_rank_mAP
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                    save_checkpoint({
                        'state_dict': state_dict,
                        }, is_best, osp.join(args.save_dir, args.arch + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))                


# python bagoftricks.py -d="prid" --name="_prid_CL_CENTERS_" --validation-training --cl-centers --print-freq=10
# python bagoftricks.py --name="_CL_CENTERS_" --validation-training --cl-centers
# python bagoftricks.py --name="_triplet_OSM_only_" --validation-training --use-OSMCAA
# python bagoftricks.py --name="_triplet_only_" --validation-training 
# python bagoftricks.py -d "ilidsvid"  --name="_ilidsvid_" --validation-training 



# Epoch 1/400 avg: 9.094356
# Epoch 2/400 avg: 8.753730
# Epoch 3/400 avg: 8.368683
# Epoch 4/400 avg: 7.851416
# Epoch 5/400 avg: 7.298941
# Epoch 6/400 avg: 6.812719
# Epoch 7/400 avg: 6.438450
# Epoch 8/400 avg: 5.961805

