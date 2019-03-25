#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  cycle_gan_model.py
#  
#  Copyright 2019 Çñè÷Ñó <Çñè÷Ñó@DESKTOP-7QS8JVL>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import torch
import itertools #???
from .base_model import BaseModel
from . import networks
from util.image_folder import ImagePool  # .query(self.fake_B)

class CycleGANModel(BaseModel):
	"""
	
	
	"""
	
	@staticmethod
	def modify_commandline_options(parser,is_train=True):
		parser.set_defaults(no_dropout=True)
		if is_train:
			parser.add_argument('--lambda_A',type = float,default = 10.0,help='weight for cycle loss()')
			parser.add_argument('--lambda_B',type=float,default=10.0,help='weight for cycle loss(B-A-B)')
			parser.add_argument('--lambda_identity',type=float,default=0.5,help='use identity mapping.Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping')
			
			
		return parser
		
	def _init_(self,opt):
		
		BaseModel._init_(self,opt)
		self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
		visual_names_A = ['real_A', 'fake_B', 'rec_A']
		visual_names_B = ['real_B', 'fake_A', 'rec_B']
		if self.isTrain and self.opt.lambda_identity > 0.0:
			visual_names_A.append('idt_B')
			visual_names_B.append('idt_A')
			
		self.visual_names = visual_names_A + visual_names_B
		
		if self.isTrain:
			self.model_names = ['G_A','G_B','D_A','D_B']
		else:
			self.model_names = ['G_A','G_B']
			
		# define networks
		self.netG_A = networks.define_G( # ´æÒÉ
		self.netG_B = networks.define_G( # ´æÒÉ
		
		
		if self.isTrain:
			self.netD_A = networks.define_D(#
			self.netD_B = networks.define_D(#
			
		if self.isTrain:
			if opt.lambda_identity > 0.0:
				assert(opt.input_nc == opt.output_nc)
			self.fake_A_pool = ImagePool(opt.pool_size)
			self.fake_B_pool = ImagePool(opt.pool_size)
			#define loss func
			self.criterionGAN = networksGANLoss(opt.gan_mode).to(self.device)
			self.criterionCycle = torch.nn.L1Loss()
			self.criterionIdt = torch.nn.L1Loss()
			# select optimizer
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(),self.netG_B.parameters()),lr = opt.lr,betas=(opt.beta1,0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),self.netD_B.parameters()),lr=opt.lr,betas=(opt.beta1,0.999))
			self.optimizer.append(self.optimizer_G)
			self.optimizer.append(self.optimizer_D)
		
	def set_input(self,input):
		AtoB = self.opt.direction == 'AtoB'
		self.real_A = input['A' if AtoB else 'B'].to(self.device)
		self.real_B = input['B' if AtoB else 'A'].to(self.device)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']
		
	def forward(self):
		
		self.fake_B = self.netG_A(self.real_A)
		self.rec_A = self.netG_B(self.fake_B)
		self.fake_A = self.netG_B(self.real_B)
		self.rec_B = self.netG_A(self.fake_A)
		
	def backward_D_basic(self.netD,real,fake):
		"""Calculate GAN loss for the discriminator
		
		"""
		pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real,True)
		
		pred_fake = netD(fake)
		loss_D_fake = self.criterionGAN(pred_fake,False)
		#combined loss and calculate gradients
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		loss_D.backward()
		
		return loss_D
		
		
		
		
	def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)


	def backward_G(self):
		"""Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
        
    def optimize_parameters(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
