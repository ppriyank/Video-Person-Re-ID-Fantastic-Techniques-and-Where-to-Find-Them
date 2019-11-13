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
| SOTA (w/o re-rank)      |   80.8  | 86.3 | 95.7 | 98.1   |
| SOTA (w/o re-rank)    |   87.7  | 87.2 | 96.2 | 98.6 |
| Baseline     |  76.7 (84.5) | 83.3 (85.0) | 93.8 (94.7) | 97.4 (97.7)|
| Baseline + BOT    |    81.3 (88.4) | 87.1 (87.6) | 95.9 (96.0) | 98.2 (98.4) |
| Baseline + BOT + OSM Loss    |  |   |   |   |
| Baseline + BOT + OSM Loss + CL Centers    |    |   |   |  |

**PRID DATASET** 
| Model            | mAP |CMC-1 | CMC-5 | CMC-20 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | 
| SOTA       |  93.2%  | -  | -  |  -   |
| Baseline + BOT + OSM Loss + CL Centers    |  93.1  |  88.8 | 97.8  | 100.0 |
| Baseline + BOT + OSM Loss + CL Centers (pretrained on MARS)   |    |   |   |  |
 


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



Ref: 

* STA: Spatial-Temporal Attention for Large-Scale Video-Based Person Re-Identification
* Diversity Regularized Spatiotemporal Attention for Video-based Person Re-identification

