#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  base_model.py
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

import os
import torch
from collections import OrderedDict
from abc import ABC,abstractmethod
from . import networks

class BaseModel(ABC):
	"""
	
	"""
	
	def _init_(self,opt):
		self.opt = opt
		self.gpu_ids = opt.isTrain
		sefl.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		self.save_dir = os.path.join(opt.checkpoints_dir,opt.name)#????
		if opt.preprocess != 'scale_width':
			torch.backends.cudnn.benchmark = True
			
		self.loss_names = []
		self.model_names = []
		self.visual_names = []
		self.optimizers = []
		self.image_paths = []
		self.metric = None # used for learning rate policy 'plateau'
		
	@staticmethod
	def modify_commandline_options(parser,is_train):
		
		return parser
		
	@abstractmethod
	def set_input(self,input):
		pass
		
	@abstractmethod
	def forward(self):
		pass
		
	@abstractmethod
	def forward(self):
		pass
		
	@abstractmethod
	def  optimize_parameters(self):
		pass
		
	def setup(self,opt):
		if self.isTrain:
			self.schedulers = [networks.get_scheduler(optimizer,opt) for optimizer in self.optimizers]
		if not self.isTrain or opt.continue_train:
			load_sufix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
			self.load_netwroks(load_suffix)
		self.print_networks(opt.verbose)
		
	def eval(self):
		
		for name in self.model_names:
			if isinstance(name,str):
				net = getattr(self,'net'+name)
				net.eval()
				
	def test(self):
		wit
		
	def compute_visuals(self):
		pass
		
	def get_image_paths(self):
		return self.image_paths
		
	def update_learning_rate(self):
		for scheduler in self.schedulers:
			scheduler.step(self.metric)
			
		lr = self.optimizers[0].param_groups[0]['lr']
		print('learning rate = %.7f' % lr)
		
	def get_current_visuals(self):
		
		visual_ret = OrderedDict()
		for name in self.visual_name:
			if isinstance(name,str):
				visual_ret[name] = getattr(self,name)
		return visual_ret
		
	def get_current_losses(self):
		""" Return training losses"""
		
		
	def save_networks(self,epoch):
		"""save all the networks to the disk"""
		
		
	def _patch_instance_norm_state_dict(self,state_dict,module,keys,i=0):
		"""Fix InstanceNorm checkpoints incompatibility """
		
		
	def print_networks(self,verbose):
		"""print the total num of paras in the network architecture
		
		para:
			verbose(bool) -- if verbose: print the network architecture
			
		"""
		
	def set_requires_grad(self,nets,requires_grad = False):
		""" Set requires_grad=False for all the networks to avoid  unnecessary computation
		
		para:
			nets(network list) -- a list of nw
			requires_grad(bool) -- if the nws require gradients or not
			"""
			



























