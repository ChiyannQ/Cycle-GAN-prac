#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  _init_.py
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

import importlib
from models.base_model import BaseModel

def find_model_using_name(model_name):
	model_filename = "models." + model_name + "_model"
	
	#import module
	modellib = importlib.import_module(model_filename)
	
	#
	model = None
	target_model_name = model_name.replace('_','') + 'model'
	
	for name,cls in modellib._dict_.item():
		if name.lower() == target_model_name.lower() \
			and issubclass(cls,BaseModel):
				model = cls
				
	if model is None:
		print("could not find the target model")
		exit(0)
	return model
	
	
	
	
	

def create_model(opt):
	
	model = find_model_using_name(opt.model)
	instance = model(opt)
	#set up?
	print("model [%s] was created" % type(instance)._name_)
	
	return instance



def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options
