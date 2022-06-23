import os
import torch
from utils import *
from options import TestOptions
from dataset import dataset_pair
from model import DerainCycleGAN
from saver import save_imgs
import os
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, Pad, ToPILImage

def main():

    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # model
    print('\n--- load model ---')
    model = DerainCycleGAN(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if opts.mode == 0:
        dataset = dataset_pair(opts, 'rainy_Rain100L', opts.input_dim_a)
    elif opts.mode == 1:
        dataset = dataset_pair(opts, 'rainy_Rain800', opts.input_dim_b)
             
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)
    
    time_test = 0
    count = 0

    transform1 = [ToPILImage(), Pad((0, 0, 3, 3), padding_mode='edge')]
    transforms1 = Compose(transform1)
    transform2 = [ToPILImage()]
    transforms2 = Compose(transform2)

    for idx1, (img1, needcrop1, img2, needcrop2) in enumerate(loader):
        print('{}/{}'.format(idx1, len(loader)))
        img1 = img1.cuda()
        img2 = img2.cuda()
        imgs = []
        names = []
        with torch.no_grad():
            img = model.test_forward(img1, img2, a2b=0)
        imgs.append(img)
        names.append('{}'.format(idx1+1))
        save_imgs(imgs, names, os.path.join(result_dir), needcrop1)

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()

