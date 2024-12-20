"""
Computes the speed of the model and plot the speed of all individual steps (converting RF lines to GPU, run the model, convert RF lines back to CPU)

Author: Rienk Zorgdrager, University of Twente, 2024
"""

# Import packages:
import torch
import os
import numpy as np
import tkinter as tk
import time

from bubbledataloader import load_dataset
from bubblenetwork import DilatedCNN
from os import listdir
import matplotlib.pyplot as plt

#%% SETTINGS
NEPOCHS = 1250
delim = '\\'
# Directory where the  RF data is stored: 


#%% LOAD THE DATA
BATCH_SIZE = 16     # Batch size
NDATA = 96

datadir = r"D:\SRML-1D-pulse-types\Results\RF signals\txt_files\pulseSingle_Reference_OneCycle\VALIDATION"
filelist = listdir(datadir)

epoch = NEPOCHS-1
model = DilatedCNN(hidden_size=64, depth=12)  
modelfile = 'epoch_' + str(epoch)
modelpath = r"D:\SRML-1D-pulse-types\Results\Networks\model_pulseSingle_Reference_OneCycle\1250_epochs\epoch_1249"
print('loaded model: ' + modelpath)

model.load_state_dict(torch.load(modelpath))
model = model.cuda()
model.eval()

# Initialize the time array
time_array = []#np.zeros([])
time_array_model = []#np.zeros([])
time_array_cuda = []#np.zeros([])
time_array_cpu = []#np.zeros([])

niters = np.arange(0,99,1)

# Compute the speed for niters and average the data
for i,j in enumerate(niters):
    ind = np.arange(0,NDATA,1)   # File indices validation data
        
    dataset = load_dataset(datadir,filelist,ind)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = BATCH_SIZE, shuffle=False)
    
    time_cuda = []
    time_cpu = []
    time_model = []
    
    t1 = time.time()
    for it, sample_batched in enumerate(dataloader):
        t1_cuda = time.time()
        V = sample_batched['x'].cuda()      # RF data
        t2_cuda = time.time()
        time_cuda.append(t2_cuda - t1_cuda)
        
        t1_model = time.time()
        z = model(V)                        # Predicted bubble distribution
        t2_model = time.time()      
        time_model.append(t2_model - t1_model) 
           
            # Convert to numpy array:  
        t1_cpu = time.time()
        z = np.squeeze(np.transpose(z.cpu().detach().numpy()))
        t2_cpu = time.time()
        time_cpu.append(t2_cpu - t1_cpu)
        
    t2 = time.time()
    tot_time = t2 - t1

    # if i>5: # The computer is slower in the first computations as it needs to activate
    time_array.append(tot_time)
    time_array_cpu.append(np.mean(time_cpu))
    time_array_cuda.append(np.mean(time_cuda))
    time_array_model.append(np.mean(time_model))
time_array = np.array(time_array)
time_array_cpu = np.array(time_array_cpu)
time_array_cuda = np.array(time_array_cuda)
time_array_model = np.array(time_array_model)
print("Elapsed time per 96 RF lines is:",time_array.mean(),"s")

#%% Plots
plt.figure()
plt.plot(niters, time_array)
plt.xlabel('iteration')
plt.ylabel('time (s)')
plt.grid()

# Plot contribution of each step
plt.figure()
tot_time = time_array_cpu + time_array_cuda + time_array_model
plt.plot(niters,np.divide(time_array_cuda,tot_time)*100, label = 'convert to CUDA')
plt.plot(niters,np.divide(time_array_cpu,tot_time)*100, label = 'convert to CPU')
plt.plot(niters,np.divide(time_array_model,tot_time)*100, label = 'Model processing')
plt.xlabel('iteration')
plt.ylabel('% of total time')
plt.ylim([0,100])
plt.legend()
plt.grid()
