import torch
from options import TestOptions
from dataset import dataset_single_test
from model import DerainCycleGAN
from saver import save_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  if opts.mode == 0:
    dataset = dataset_single_test(opts, '_rain100H', opts.input_dim_a)
  elif opts.mode == 1:
    dataset = dataset_single_test(opts, '_rain100L', opts.input_dim_b)
  elif opts.mode == 2:
    dataset = dataset_single_test(opts, '_rain12', opts.input_dim_b)  
  elif opts.mode == 3:
    dataset = dataset_single_test(opts, '_real', opts.input_dim_b)      
  elif opts.mode == 4:
    dataset = dataset_single_test(opts, '_rain800', opts.input_dim_b)  
  elif opts.mode == 5:
    dataset = dataset_single_test(opts, '_SPA', opts.input_dim_b)  
  elif opts.mode == 6:
    dataset = dataset_single_test(opts, '_practical', opts.input_dim_b)       
  elif opts.mode == 7:
    dataset = dataset_single_test(opts, '_real', opts.input_dim_b)                  
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

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

  # test
  print('\n--- testing ---')
  for idx1, (img1, needcrop) in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img1 = img1.cuda()
    imgs = []
    names = []
    for idx2 in range(1):
      with torch.no_grad():
        img = model.test_forward(img1, a2b=opts.a2b)
      imgs.append(img)
      names.append('{}'.format(idx1+1))
    save_imgs(imgs, names, os.path.join(result_dir), needcrop)

  return

if __name__ == '__main__':
  main()
