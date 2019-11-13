# Video Person Re-ID: Fantastic Techniques and Where to Find Them
Official Pytorch Implementation of the paper:  
**Video Person Re-ID: Fantastic Techniques and Where to Find Them** *(Submitted in AAAI'20)*  
*Priyank Pathak,  Amir Erfan Eshratifar,  Michael Gormish*   
*(Work done in collaboration of NYU and Clarifai)*



Extension of Work done for Videos 
* [Revisiting Temporal Modeling for Video-based Person ReID](https://github.com/jiyanggao/Video-Person-ReID)
* [Bag of Tricks and A Strong ReID Baseline](https://github.com/michuanhaohao/reid-strong-baseline)
* [Online-Soft-Mining-and-Class-Aware-Attention](https://github.com/ppriyank/-Online-Soft-Mining-and-Class-Aware-Attention-Pytorch)



## RESULTS

### Performance

**MARS DATASET** 

| Model            | mAP |CMC-1 | CMC-5 | CMC-20 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | 
| SOTA (w/ re-rank)      |   80.8  | 86.3 | 95.7 | 98.1   |
| SOTA (w/o re-rank)    |   75.8  | 83.1 | 92.8 | 95.3 |
| Baseline     |  76.7 | 83.3 | 93.8 | 96.0 |
| Baseline + BOT    |   73.9 | 81.6 | 92.8 | 94.7 |
| Baseline + BOT + OSM Loss    |  70.5 | 78.5 | 90.9 | 93.9 |
| Baseline + BOT + OSM Loss + CL Centers    |  70.5 | 78.5 | 90.9 | 93.9 |



## MODEL

<img src="https://github.com/ppriyank/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them/blob/master/images/diag.jpg" width="900">


## DATASET

MARS dataset: 

<img src="https://github.com/ppriyank/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them/blob/master/images/data.jpg" width="400">





## bag of tricks   
`args.arch = "ResNet50ta_bt"`

`python bagoftricks.py --name="_CL_CENTERS_" --validation-training --cl-centers`  
`python bagoftricks.py --name="_triplet_OSM_only_" --validation-training --use-OSMCAA`  
`python bagoftricks.py --name="_triplet_only_" --validation-training`   
`python bagoftricks.py --name="_ilidsvid_" --validation-training`   
`python bagoftricks.py --name="_prid_" --validation-training`   

## hyper parameter optimization   

`python hyper_supervise_validation.py --focus="map"`       
`python hyper_supervise_validation.py --focus="rerank_map" `      
`python hyper_supervise_validation.py --focus="map" --sampling="inteliigi"`      
`python hyper_supervise_validation.py --focus="rerank_map" --sampling="inteliigi"`    
