# DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking (TIP2021)
Yanyan Wei, Zhao Zhang, Yang Wang, Mingliang Xu, Yi Yang, Shuicheng Yan, Meng Wang

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
