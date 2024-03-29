# DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking (TIP 2021)
Yanyan Wei, Zhao Zhang, Yang Wang, Mingliang Xu, Yi Yang, Shuicheng Yan, Meng Wang

### Update
2023.01.23: Upload the pre-trained models.

### Abstract
Single Image Deraining (SID) is a relatively new and still challenging topic in emerging vision applications, and most of the recently emerged deraining methods use the supervised manner depending on the ground-truth (i.e., using paired data). However, in practice it is rather common to encounter unpaired images in real deraining task. In such cases, how to remove the rain streaks in an unsupervised way will be a challenging task due to lack of constraints between images and hence suffering from low-quality restoration results. In this paper, we therefore explore the unsupervised SID issue using unpaired data, and propose a new unsupervised framework termed DerainCycleGAN for single image rain removal and generation, which can fully utilize the constrained transfer learning ability and circulatory structures of CycleGAN. In addition, we design an unsupervised rain attentive detector (UARD) for enhancing the rain information detection by paying attention to both rainy and rain-free images. Besides, we also contribute a new synthetic way of generating the rain streak information, which is different from the previous ones. Specifically, since the generated rain streaks have diverse shapes and directions, existing derianing methods trained on the generated rainy image by this way can perform much better for processing real rainy images. Extensive experimental results on synthetic and real datasets show that our DerainCycleGAN is superior to current unsupervised and semi-supervised methods, and is also highly competitive to the fully-supervised ones.

![image](https://github.com/OaDsis/DerainCycleGAN/blob/main/figures/model.png)
![image](https://github.com/OaDsis/DerainCycleGAN/blob/main/figures/result.png)

### Requirements
- python 3.6.10
- torch 1.4.0
- torchvision 0.5.0
- NVIDIA GeForce GTX GPU with 12GB memory at least, or you can change image size in option.py

### Datasets
- Rain100L
- Rain800
- Rain12
- SPA-Data
- Real-Data

You can download above datasets from [here](https://github.com/hongwang01/Video-and-Single-Image-Deraining#datasets-and-discriptions)

### Pre-trained Models
You can download pre-trained models from [here](https://drive.google.com/drive/folders/1DvOFGIdXXnNm1iage69HuasUPHZSXYYt?usp=sharing) and put them into corresponding folders, then the content is just like:

"./results/Rain100L/net_best_Rain100L.pth"

"./results/Rain800/net_best_Rain800.pth"

"./vgg16/vgg16.weight"

Note: **net_best_Rain100L.pth** is for the testing of Rain100L, Rain12, SPA-Data, and Real-Data datasets. **net_best_Rain800.pth** is for the testing of Rain800 dataset. **vgg16.weight** is for the parameters of Vgg16.

### Usage
#### Prepare dataset:
Taking training Rain100L as example. Download Rain100L (including training set and testing set) and put them into the folder "./datasets", then the content is just like:

"./datasets/rainy_Rain100L/trainA/rain-***.png"

"./datasets/rainy_Rain100L/trainB/norain-***.png"

"./datasets/test_rain100L/trainA/rain-***.png"

"./datasets/test_rain100L/trainB/norain-***.png"
#### Train (Take Rain100L dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/rainy_Rain100L --val_path ../datasets/test_rain100L --name Rain100L
```
#### Test (Take Rain100L dataset as example):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py --test_path ../datasets --name Rain100L --resume ../results/Rain100L/net_best_Rain100L.pth --mode 1
```
you can change the mode to test different datasets, i.e., Rain100L = 1, Rain12 = 2, Real-Data = 3, Rain800 = 4, SPA-Data = 5.
#### Generate Rain Images
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 auto.py --auto_path ../datasets --name Auto100L --resume ../results/Rain100L/net_best_Rain100L.pth --mode 0 --a2b 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 auto.py --auto_path ../datasets --name Auto800 --resume ../results/Rain800/net_best_Rain800.pth --mode 1 --a2b 0
```
### Citation
Please cite our paper if you find the code useful for your research.
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

### Contact
Thanks for your attention. If you have any questions, please contact my email: weiyy@hfut.edu.cn. 
