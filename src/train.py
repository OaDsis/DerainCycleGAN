import torch
from options import TrainOptions
from dataset import dataset_unpair, dataset_unpair_val
from model import DerainCycleGAN
from saver import Saver
import os
from SSIM import *
from utils import *
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_unpair(opts)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
  dataset_val = dataset_unpair_val(opts)
  loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=opts.nThreads)     
  criterion = SSIM()
  criterion.cuda(opts.gpu)
  if not os.path.exists(os.path.join(opts.result_dir, opts.name)):
    os.makedirs(os.path.join(opts.result_dir, opts.name))
  trainLogger = open('%s/psnr&ssim.log' % os.path.join(opts.result_dir, opts.name), 'w')

  # model
  print('\n--- load model ---')
  model = DerainCycleGAN(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)

  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  for ep in range(ep0, opts.n_ep):

    ssim_sum=0
    ssim_avg=0
    psnr_sum=0
    psnr_avg=0    

    for it, (images_a, images_b) in enumerate(train_loader):
      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue

      # input data
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()

      model.update_EG(images_a, images_b, ep, opts)
      model.update_D(opts)
      
      # save to display file
      if not opts.no_display_img:
        saver.write_display(total_it, model)

      print('total_it: %d (ep %d, it %d), lr %08f, disA %04f, disB %04f, ganA %04f, ganB %04f, recA %04f, recB %04f, percp %04f, cons_loss %04f, attB %04f, total %04f' %  (total_it, ep, it, model.genA_opt.param_groups[0]['lr'], \
                                                      model.disA_loss, model.disB_loss, \
                                                      model.gan_loss_a, model.gan_loss_b, \
                                                      model.l1_recon_A_loss, model.l1_recon_B_loss, \
                                                      model.perceptual_loss, model.cons_loss, model.atten_loss_b, model.G_loss))
      total_it += 1

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)
    
    print('\n--- valing ---')
    model.eval()
    for i, (input_val, target_val) in enumerate(loader_val, 0):        
        
        input_val, target_val = input_val.cuda(opts.gpu), target_val.cuda(opts.gpu)  
        out_val = model.test_forward(input_val, a2b=opts.a2b)        
        ssim_val = criterion(target_val, out_val)
        ssim_sum = ssim_sum + ssim_val.item()
        out_val = torch.clamp(out_val, 0., 1.)
        psnr_val = batch_PSNR(out_val, target_val, 1.) 
        psnr_sum = psnr_sum + psnr_val

        print("[epoch %d][%d/%d] ssim: %.4f, psnr: %.4f" %
              (ep+1, i+1, len(loader_val), ssim_val.item(), psnr_val))  
                          
    ssim_avg = ssim_sum/len(loader_val)
    psnr_avg = psnr_sum/len(loader_val)

    trainLogger.write('%03d\t%04f\t%04f\r\n' % \
                  (ep, psnr_avg, ssim_avg))
    trainLogger.flush()


    if ep == ep0:
        best_psnr = psnr_avg
        best_ssim = ssim_avg

    print("[epoch %d][%d/%d] ssim_avg: %.4f, psnr_avg: %.4f, best_ssim: %.4f, best_psnr: %.4f" %
            (ep+1, i+1, len(loader_val), ssim_avg, psnr_avg, best_ssim, best_psnr)) 

    if (ssim_avg >= best_ssim) and (psnr_avg >= best_psnr):
        best_psnr = psnr_avg
        best_ssim = ssim_avg         
        print('--- save the model @ ep %d ---' % (ep))
        model.save('%s/net_best_%05d.pth' % (os.path.join(opts.result_dir, opts.name), ep), ep, total_it)
        
  trainLogger.close()

  return

if __name__ == '__main__':
  main()
