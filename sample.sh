#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --time=15:00:00
#SBATCH --mem=100000
#SBATCH --job-name=pp1953
#SBATCH --mail-user=pp1953p@nyu.edu
#SBATCH --output=output/slurm_%j.out


. ~/.bashrc
module load anaconda3/5.3.1

conda activate PPUU
conda install -n PPUU nb_conda_kernels

chikka=$1

echo $chikka


cd 
cd /home/pp1953/code/official
# python hyper_supervise_validation.py --focus="map" --dataset="prid_subset" >> ~/code/Personre-id-master/outputs/prid_diff_val_map.out
# python hyper_supervise_validation.py  --dataset="ilidsvid_subset" >> ~/code/Personre-id-master/outputs/ilvid_subset_rand_bt6.out

# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset" --sampling="intelligent" >> ~/code/temp/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them-master/output/intelligent_mars_cl_centers.out
# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset" --sampling="intelligent" >> ~/code/temp/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them-master/output/intelligent_mars_cl_centers.out
# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset2" --sampling="intelligent" >> ~/code/official/output/subset2_intelligent_mars_cl_centers.out
# python hyper_supervise_validation.py  --focus="map" --dataset="mars_subset2"  >> ~/code/official/output/subset2_random_mars_cl_centers.out
python config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_cl_centers_ --cl-centers >>  ~/code/official/output/mars_cl_centers_$chikka.out
# python3 config_trainer.py --focus=map --dataset=mars --opt=2 --name=_mars_cl_centers_ --cl-centers >> output/mars_cl_centers_2.out
