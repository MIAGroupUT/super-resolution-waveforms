# -*- coding: utf-8 -*-
"""
This script defines an undilated CNN, which is used to compute the number of weights and biases.

Rienk Zorgdrager, University of Twente, 2024
"""

import torch
import torch.nn as nn

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