# DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking (TIP 2021)
Yanyan Wei, Zhao Zhang, Yang Wang, Mingliang Xu, Yi Yang, Shuicheng Yan, Meng Wang

### Abstract
Single Image Deraining (SID) is a relatively new and still challenging topic in emerging vision applications, and most of the recently emerged deraining methods use the supervised manner depending on the ground-truth (i.e., using paired data). However, in practice it is rather common to encounter unpaired images in real deraining task. In such cases, how to remove the rain streaks in an unsupervised way will be a challenging task due to lack of constraints between images and hence suffering from low-quality restoration results. In this paper, we therefore explore the unsupervised SID issue using unpaired data, and propose a new unsupervised framework termed DerainCycleGAN for single image rain removal and generation, which can fully utilize the constrained transfer learning ability and circulatory structures of CycleGAN. In addition, we design an unsupervised rain attentive detector (UARD) for enhancing the rain information detection by paying attention to both rainy and rain-free images. Besides, we also contribute a new synthetic way of generating the rain streak information, which is different from the previous ones. Specifically, since the generated rain streaks have diverse shapes and directions, existing derianing methods trained on the generated rainy image by this way can perform much better for processing real rainy images. Extensive experimental results on synthetic and real datasets show that our DerainCycleGAN is superior to current unsupervised and semi-supervised methods, and is also highly competitive to the fully-supervised ones.

![image](https://github.com/OaDsis/DerainCycleGAN/blob/main/figures/model.png)

### Requirements
- python 3.6.10
- torch 1.4.0
- torchvision 0.5.0

### Datasets
- Rain100L
- Rain800
- Rain12
- SPA-Data

### Usage
#### Prepare dataset:
Taking training Rain100L as an example. Download Rain100L (including training set and testing set) and put them into the folder "./datasets", then the content is just like:

"./datasets/Rain100L_train/trainA/rain-***.png"

"./datasets/Rain100L_train/trainB/norain-***.png"

"./datasets/Rain100L_test/trainA/rain-***.png"

"./datasets/Rain100L_test/trainB/norain-***.png"
#### Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/Rain100L_train --val_path ../datasets/Rain100L_test --name TEST
```
#### Test:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py --test_path ../datasets --name TEST --resume ../results/TEST/net_best_*****.pth --mode 1
```
### Citation
```
@article{wei2021deraincyclegan,
  title={Deraincyclegan: Rain attentive cyclegan for single image deraining and rainmaking},
  author={Wei, Yanyan and Zhang, Zhao and Wang, Yang and Xu, Mingliang and Yang, Yi and Yan, Shuicheng and Wang, Meng},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={4788--4801},
  year={2021},
  publisher={IEEE}
}
```
### Acknowledgement
Code borrows from [DRIT](https://github.com/HsinYingLee/DRIT) by Hsin-Ying Lee. Thanks for sharing !
