# Decouple-and-Couple Learning in Multi-Modal Brain Tumor Segmentation [IEEE JBHI'25]


## Introduction

Exploiting multi-modal magnetic resonance imaging complementary information for brain tumor segmentation is still a challenging task. Existing methods are usually inclined to learn the joint representation of all tumor regions indiscriminately, thus salient sub-region or healthy tissue would be dominant during the training procedure, which leads to a biased and limited representation performance. In this study, a novel transformer-based multi-modal brain tumor segmentation approach is developed by decoupling and coupling strategy. First, Anatomy-induced Region Decoupler decouples the representation of the tumor scattered in different semantic sub-regions following anatomical view, which forces the model to fully learn intra-region representation separately with multiple modalities context. Additionally, we introduce the collaborative decoupling of the corresponding sub-region edge to serve auxiliary cues. We then design the Edge-supported Intra-region Coupler to separately couple edge and object learning within each anatomical sub-region structure. Lastly, the Mutual Cross-region Coupler is further applied to implement mutual improvement by coupling complementary gains among the above decoupled sub-regions. Extensive experiments clearly demonstrate that our method outperforms current state-of-the-arts for brain tumor segmentation on BRATS2018, BRATS2020, MSD, and BRATS2021 benchmarks while retaining high efficiency in the learning procedure.


## Update

2025/2: the code released.

## Usage

1. Install pytorch 

   - The code is tested on python 3.7 and pytorch 1.2.0.

2. Dataset
   
   You can download original datasets:
   - BRAST 2018, 2020, 2021: 
   - MSD: 

## Reference

If you consider use this code, please cite our paper:

```
@article{wang2025decouple,
  title={Decouple-and-Couple Learning in Multi-Modal Brain Tumor Segmentation},
  author={Xiao, Fuan and Ji, Chaojie and Zhang, zheng and Wang, Ruxin},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

     
 

