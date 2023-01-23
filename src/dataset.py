import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, Pad, ToPILImage
import random
import torch.nn.functional as F
import cv2
import pdb

class dataset_single_test(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.test_path = opts.test_path
    images = os.listdir(os.path.join(self.test_path, opts.phase + setname, 'trainA'))
    self.img = [os.path.join(self.test_path, opts.phase + setname, 'trainA', x) for x in images]   
    self.img.sort()
    self.size = len(self.img)
    self.input_dim = input_dim
    self.img_name = self.img
    transforms1 = [Pad((0, 0, 3, 3), padding_mode='edge'), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms1 = Compose(transforms1)
    transforms2 = [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms2 = Compose(transforms2)    
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data, needcrop = self.load_img(self.img[index], self.input_dim)
    return data, needcrop

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    y = cv2.imread(img_name)
    h,w = y.shape[0], y.shape[1]
    needcrop = 0
    if h == 321 or w == 321:
      img = self.transforms1(img)
      needcrop = 1
    else:
      img = self.transforms2(img)    
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img, needcrop

  def __len__(self):
    return self.size

class dataset_pair(data.Dataset):
  def __init__(self, opts, name, input_dim):
    self.auto_path = opts.auto_path
    # A
    images_A = os.listdir(os.path.join(self.auto_path, name, 'trainA'))
    self.A = [os.path.join(self.auto_path, name, 'trainA', x) for x in images_A]
    self.A.sort()
    # B
    images_B = os.listdir(os.path.join(self.auto_path, name, 'trainB'))
    self.B = [os.path.join(self.auto_path, name, 'trainB', x) for x in images_B]
    self.B.sort()
    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    transforms1 = [Pad((0, 0, 3, 3), padding_mode='edge'), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms1 = Compose(transforms1)
    transforms2 = [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms2 = Compose(transforms2)    
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A, needcrop1 = self.load_img(self.A[index], self.input_dim_A)
      data_B, needcrop2 = self.load_img(self.B[index], self.input_dim_B)
    else:
      data_A, needcrop1 = self.load_img(self.A[index], self.input_dim_A)
      data_B, needcrop2 = self.load_img(self.B[index], self.input_dim_B)
    return data_A, needcrop1, data_B, needcrop2

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    y = cv2.imread(img_name)
    h,w = y.shape[0], y.shape[1]
    needcrop = 0
    if h == 321 or w == 321:
      img = self.transforms1(img)
      needcrop = 1
    else:
      img = self.transforms2(img)    
    return img, needcrop

  def __len__(self):
    return self.dataset_size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.train_path = opts.train_path
    # A
    images_A = os.listdir(os.path.join(self.train_path, opts.phase + 'A'))
    self.A = [os.path.join(self.train_path, opts.phase + 'A', x) for x in images_A]
    # B
    images_B = os.listdir(os.path.join(self.train_path, opts.phase + 'B'))
    self.B = [os.path.join(self.train_path, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

class dataset_unpair_val(data.Dataset):
  def __init__(self, opts):
    self.val_path = opts.val_path

    # A
    images_A = os.listdir(os.path.join(self.val_path, opts.phase + 'A'))
    self.A = [os.path.join(self.val_path, opts.phase + 'A', x) for x in images_A]
    # B
    images_B = os.listdir(os.path.join(self.val_path, opts.phase + 'B'))
    self.B = [os.path.join(self.val_path, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Pad((0, 0, 3, 3), padding_mode='edge'), CenterCrop(64), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]    
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size