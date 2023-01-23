import networks
import torch
import torch.nn as nn
import os
import pickle
from utils import *

class DerainCycleGAN(nn.Module):
  def __init__(self, opts):
    super(DerainCycleGAN, self).__init__()

    # parameters
    lr = 0.0001
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    # discriminators        
    self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)

    # urad 
    self.urad = networks.URAD()

    # generator
    self.genA = networks.Gen()
    self.genB = networks.Gen()      

    # vgg
    self.vgg = networks.Vgg16()
    networks.init_vgg16('../vgg16/')
    self.vgg.load_state_dict(torch.load(os.path.join('../vgg16/', "vgg16.weight")))

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.urad_opt = torch.optim.Adam(self.urad.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)    
    self.genA_opt = torch.optim.Adam(self.genA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.genB_opt = torch.optim.Adam(self.genB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    self.criterionL2 = torch.nn.MSELoss()
    self.criterionGAN = GANLoss(opts.gan_mode).cuda(opts.gpu)    

    # create image buffer to store previously generated images
    self.fake_A_pool = ImagePool(opts.pool_size)  
    self.fake_B_pool = ImagePool(opts.pool_size) 
    
  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.urad.apply(networks.gaussian_weights_init)
    self.genA.apply(networks.gaussian_weights_init)
    self.genB.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.urad_sch = networks.get_scheduler(self.urad_opt, opts, last_ep)
    self.genA_sch = networks.get_scheduler(self.genA_opt, opts, last_ep)
    self.genB_sch = networks.get_scheduler(self.genB_opt, opts, last_ep)    

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.urad.cuda(self.gpu)
    self.genA.cuda(self.gpu)
    self.genB.cuda(self.gpu)    
    self.vgg.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):                     # 
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_forward(self, image1, image2=None, a2b=None):
    if a2b:
        self.mask_a = self.urad(image1)
        self.frame1_a, self.frame2_a, self.fake_A_encoded = self.genA.forward(image1, self.mask_a[5])  
    else:
        self.mask_a = self.urad(image1)
        batch_size, row, col = self.mask_a[5].size(0), self.mask_a[5].size(2), self.mask_a[5].size(3)
        noise = (torch.rand(batch_size, 1, row, col) * 0.01).cuda()
        self.mask_a[5] = self.mask_a[5] + noise
        self.frame1_b, self.frame2_b, self.fake_A_encoded = self.genB.forward(image2, self.mask_a[5])  
    return self.fake_A_encoded

  def forward(self, ep, opts):
    '''self.real_A_encoded -> self.fake_A_encoded -> self.real_A_recon'''
    '''self.real_B_encoded -> self.fake_B_encoded -> self.real_B_recon'''    
    # input images
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A    
    self.real_B_encoded = real_B   

    # get first cycle
    '''self.real_A_encoded -> self.fake_A_encoded'''
    '''self.real_B_encoded -> self.fake_B_encoded'''    
    self.mask_a = self.urad(self.real_A_encoded)
    self.frame1_a, self.frame2_a, self.fake_A_encoded = self.genA.forward(self.real_A_encoded, self.mask_a[5])  
    self.mask_b = self.urad(self.real_B_encoded)
    self.frame1_b, self.frame2_b, self.fake_B_encoded = self.genB.forward(self.real_B_encoded, self.mask_b[5]) 

    # get perceptual loss
    self.perc_real_A = self.vgg(self.real_A_encoded).detach()
    self.perc_fake_A = self.vgg(self.fake_A_encoded).detach()

    # get second cycle
    '''self.fake_A_encoded -> self.real_A_recon'''
    '''self.fake_B_encoded -> self.real_B_recon'''    
    self.mask_b2 = self.urad(self.fake_B_encoded)
    self.frame1_b2, self.frame2_b2, self.real_B_recon = self.genA.forward(self.fake_B_encoded, self.mask_b2[5])  
    self.mask_a2 = self.urad(self.fake_A_encoded)
    self.frame1_a2, self.frame2_a2, self.real_A_recon = self.genB.forward(self.fake_A_encoded, self.mask_a2[5]) 

    self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                    self.real_A_recon[0:1].detach().cpu(), \
                                    self.real_B_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                    self.real_B_recon[0:1].detach().cpu()), dim=0)

  def update_D(self, opts):
    self.fake_A_encoded = self.fake_A_pool.query(self.fake_A_encoded)
    self.fake_B_encoded = self.fake_B_pool.query(self.fake_B_encoded)
    
    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D_basic(self.disA, self.real_A_encoded, self.fake_B_encoded)    
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D_basic(self.disB, self.real_B_encoded, self.fake_A_encoded)    
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

  def backward_D_basic(self, netD, real, fake):
      # Real
      pred_real = netD(real)
      loss_D_real1 = self.criterionGAN(pred_real[0], True)
      loss_D_real2 = self.criterionGAN(pred_real[1], True)
      loss_D_real3 = self.criterionGAN(pred_real[2], True)    
      loss_D_real = (loss_D_real1 + loss_D_real2 + loss_D_real3) / 3   

      # Fake
      pred_fake = netD(fake.detach())
      loss_D_fake1 = self.criterionGAN(pred_fake[0], False)
      loss_D_fake2 = self.criterionGAN(pred_fake[1], False)
      loss_D_fake3 = self.criterionGAN(pred_fake[2], False)  
      loss_D_fake = (loss_D_fake1 + loss_D_fake2 + loss_D_fake3) / 3      

      loss_D = (loss_D_real + loss_D_fake) * 0.5
      loss_D.backward()
      return loss_D

  def update_EG(self, image_a, image_b, ep, opts):
    self.input_A = image_a
    self.input_B = image_b    
    self.forward(ep, opts)
    
    self.urad_opt.zero_grad()
    self.genA_opt.zero_grad()    
    self.genB_opt.zero_grad()    
    self.backward_EG(opts)
    self.urad_opt.step()
    self.genA_opt.step()
    self.genB_opt.step()    

  def backward_EG(self, opts):
    # adversarial loss
    loss_G_GAN_A = self.criterionGAN(self.disA(self.fake_B_encoded)[0], True)
    loss_G_GAN_B = self.criterionGAN(self.disB(self.fake_A_encoded)[0], True)  

    # cross cycle consistency loss
    loss_G_L1_A = self.criterionL1(self.real_A_recon, self.real_A_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.real_B_recon, self.real_B_encoded) * 10

    # perceptual loss
    loss_perceptual = self.criterionL2(self.perc_fake_A, self.perc_real_A) * 0.01

    # atten loss
    # cha1, row1, col1 = self.mask_a[3].shape[1], self.mask_a[3].shape[2], self.mask_a[3].shape[3]
    # mask_a_gt = torch.rand(opts.batch_size, cha1, row1, col1).cuda() 
    # loss_att_a = self.criterionL2(self.mask_a[5], mask_a_gt) * 10

    cha2, row2, col2 = self.mask_b[3].shape[1], self.mask_b[3].shape[2], self.mask_b[3].shape[3]
    mask_b_gt = torch.zeros(opts.batch_size, cha2, row2, col2).cuda() 
    loss_att_b = self.criterionL2(self.mask_b[5], mask_b_gt) * 10

    # Consistency loss
    fake = self.mask_a[5] + self.fake_A_encoded
    loss_cons = self.criterionL2(fake, self.real_A_encoded) * 10   
  
    loss_G = loss_G_GAN_A + loss_G_GAN_B + \
             loss_G_L1_A + loss_G_L1_B + \
             loss_perceptual + \
             loss_cons + \
             loss_att_b

    loss_G.backward(retain_graph=True)

    self.gan_loss_a = loss_G_GAN_A.item()
    self.gan_loss_b = loss_G_GAN_B.item()
    self.l1_recon_A_loss = loss_G_L1_A.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.perceptual_loss = loss_perceptual.item()
    # self.atten_loss_a = loss_att_a.item()    
    self.atten_loss_b = loss_att_b.item()
    self.cons_loss = loss_cons.item()   
    self.G_loss = loss_G.item()

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.urad_sch.step()
    self.genA_sch.step()
    self.genB_sch.step()

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disB.load_state_dict(checkpoint['disB'])
    self.urad.load_state_dict(checkpoint['atten'])
    self.genA.load_state_dict(checkpoint['genA'])
    self.genB.load_state_dict(checkpoint['genB'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.urad_opt.load_state_dict(checkpoint['atten_opt'])
      self.genA_opt.load_state_dict(checkpoint['genA_opt'])
      self.genB_opt.load_state_dict(checkpoint['genB_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'disA': self.disA.state_dict(),
             'disB': self.disB.state_dict(),
             'atten': self.urad.state_dict(),
             'genA': self.genA.state_dict(),
             'genB': self.genB.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'atten_opt': self.urad_opt.state_dict(),
             'genA_opt': self.genA_opt.state_dict(),
             'genB_opt': self.genB_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def save_dict(self, obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

  def load_dict(self, name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)    

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a3 = self.normalize_image(self.real_A_recon).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b3 = self.normalize_image(self.real_B_recon).detach()
    images_mask_a1 = self.normalize_image(self.mask_a[0]).detach()
    images_mask_a2 = self.normalize_image(self.mask_a[1]).detach()
    images_mask_a3 = self.normalize_image(self.mask_a[2]).detach()
    images_mask_a4 = self.normalize_image(self.mask_a[3]).detach()
    images_mask_a5 = self.normalize_image(self.mask_a[4]).detach()
    images_mask_a6 = self.normalize_image(self.mask_a[5]).detach()

    images_mask_b1 = self.normalize_image(self.mask_b[0]).detach()
    images_mask_b2 = self.normalize_image(self.mask_b[1]).detach()
    images_mask_b3 = self.normalize_image(self.mask_b[2]).detach()
    images_mask_b4 = self.normalize_image(self.mask_b[3]).detach()
    images_mask_b5 = self.normalize_image(self.mask_b[4]).detach()
    images_mask_b6 = self.normalize_image(self.mask_b[5]).detach()

    row1 = torch.cat((images_a[0:1, ::], images_a1[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_b1[0:1, ::], images_b3[0:1, ::]),3)    
    row3 = torch.cat((images_mask_a1[0:1, ::], images_mask_a2[0:1, ::], images_mask_a3[0:1, ::]),3) 
    row4 = torch.cat((images_mask_a4[0:1, ::], images_mask_a5[0:1, ::], images_mask_a6[0:1, ::]),3)    
    row5 = torch.cat((images_mask_b1[0:1, ::], images_mask_b2[0:1, ::], images_mask_b3[0:1, ::]),3) 
    row6 = torch.cat((images_mask_b4[0:1, ::], images_mask_b5[0:1, ::], images_mask_b6[0:1, ::]),3)    
    return torch.cat((row1,row2),2), torch.cat((row3,row4), 2), torch.cat((row5,row6), 2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]