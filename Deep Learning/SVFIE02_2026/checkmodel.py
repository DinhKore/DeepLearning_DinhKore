# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
from models.modelBT1 import *

model = NetBT1()

# model = model.cuda() // chạy bằng GPU nếu có, nếu không sẽ chạy bằng CPU
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
#summary(model, (3, 32, 32))