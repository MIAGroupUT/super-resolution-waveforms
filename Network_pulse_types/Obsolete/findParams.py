# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:53:18 2024

@author: rienk
"""
import torch
import torch.nn as nn
import numpy as np
import re

from bubblenetwork import DilatedCNN
# from undilatedCNN import CNN

class CNN(torch.nn.Module):
    def __init__(self, hidden_size=64, depth=4095):
        super(CNN, self).__init__()
        model = []
        model += [nn.ReflectionPad1d(2**depth - 1)]
        model += [nn.Conv1d(1, hidden_size, kernel_size=3, dilation=1)]
        model += [nn.BatchNorm1d(hidden_size)]
        model += [nn.ReLU()]      
        for l in range(1, depth):
            model += [nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1)]
            model += [nn.BatchNorm1d(hidden_size)]
            model += [nn.ReLU()]    
        model += [nn.Conv1d(hidden_size, 1, kernel_size=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        z = self.model(x)

        return z

delim = '\\'

torch.cuda.empty_cache()

modeldir = 'D:\\SRML-1D-pulse-types\\Results\\Networks\\model_pulseSingle_Reference_OneCycle\\1250_epochs'

model = DilatedCNN(hidden_size=64, depth=12)  

modelpath = modeldir + delim + "epoch_" + str(1249)
model.load_state_dict(torch.load(modelpath))

# Create a list with the layers
layers = []

for name, module in model.model._modules.items():
    layers.append(module)

info_dict = {}
i = 0

for name, param in model.named_parameters():
    modelnum = [int(s) for s in re.findall(r'\b\d+\b',name)]

    
    d = {'module':layers[modelnum[0]],'data':param.data,'numel':torch.numel(param.data)}
    info_dict[name] = d
    
    i = i+1
    

# Compute number of weights and biases
nWeights = 0
nBiases = 0

for key in info_dict.keys():
    if 'weight' in key:
        nw = info_dict[key]['numel']
        nWeights = nWeights + nw
    elif 'bias' in key:
        nb = info_dict[key]['numel']
        nBiases = nBiases + nb

print('Dilated CNN has {} weights and {} biases'.format(nWeights,nBiases))

#%% Do the same for the undilated
    
    
model = CNN(hidden_size=64, depth=4095)

# Create a list with the layers
layers = []

for name, module in model.model._modules.items():
    layers.append(module)

info_dict = {}
i = 0

for name, param in model.named_parameters():
    modelnum = [int(s) for s in re.findall(r'\b\d+\b',name)]

    
    d = {'module':layers[modelnum[0]],'data':param.data,'numel':torch.numel(param.data)}
    info_dict[name] = d
    
    i = i+1
    

# Compute number of weights and biases
nWeights = 0
nBiases = 0

for key in info_dict.keys():
    if 'weight' in key:
        nw = info_dict[key]['numel']
        nWeights = nWeights + nw
    elif 'bias' in key:
        nb = info_dict[key]['numel']
        nBiases = nBiases + nb
print('Undilated CNN has {} weights and {} biases'.format(nWeights,nBiases))