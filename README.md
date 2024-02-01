# SAL

This is the source code accompanying the paper [***How Does Unlabeled Data Provably Help Out-of-Distribution Detection?***](https://openreview.net/forum?id=jlEjB8MVGa) by Xuefeng Du, Zhen Fang, Ilias Diakonikolas, and Yixuan Li



## Ads 

Check out our ICML'23 [SCONE](https://github.com/deeplearning-wisc/scone) on using wild data for both OOD detection and generalization if you are interested!



## Dataset Preparation


**CIFAR-10/CIFAR-100**

* The dataloader will download it automatically when first running the programs.

**OOD datasets**


* The OOD datasets with CIFAR-100 as in-distribution are 5 OOD datasets, i.e., SVHN, PLACES365, LSUN-C, LSUN-R, TEXTURES.
* Please refer to Part 1 and 2 of the codebase [here](https://github.com/deeplearning-wisc/knn-ood). 





## Training and Inference

Please execute the following in the command shell:
```
python main.py --dataset cifar 10 --aux_out_dataset lsun_c --test_out_dataset lsun_c --pi 0.1 --num_class 10
```
"dataset" denotes the in-distribution training data.

"aux_out_dataset" determines the type of OOD data in the unlabeled wild data



## Limitations
Calculating the score is kind of slow for a large unlabeled wild data right now. I will need to think about how to speed up this procedure. Please consider use a small OOD dataset to construct the wild data, such as Textures/LSUN-C/LSUN-R for quick verification.


## Citation ##
If you found any part of this code is useful in your research, please consider citing our paper:

```
@inproceedings{du2024sal,
  title={How Does Wild Data Provably Help OOD Detection?},
  author={Du, Xuefeng and Fang, Zhen and  Diakonikolas, Ilias and Li, Yixuan},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2024}
}
```





