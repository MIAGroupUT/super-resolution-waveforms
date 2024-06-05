# -*- coding: utf-8 -*-

import torch
import numpy as np
from os import listdir, path, makedirs
import matplotlib.pyplot as plt

from bubbledataloader import load_dataset
from bubblenetwork import DilatedCNN
from customModelInfo import model_info

torch.cuda.empty_cache()

delim = "\\"

model_list = np.array(list(model_info.keys()))

# Filter if needed
model_filter = ''#'compressed'

if len(model_filter)>0:
    idx_filter = [model_list[k].__contains__(model_filter) for k,model in enumerate(model_list)]
else:
    idx_filter = np.arange(0,len(model_list),dtype='int32')
    
model_list = list(model_list[idx_filter])

NEPOCHS = 1250


for k,modelname in enumerate(model_list):
    print(modelname)
    model_properties = model_info[modelname]
    
    #%% FILE DIRECTORIES
    
    datadir = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + modelname + "\\TRAINING"
    filelist = listdir(datadir)
    
    modeldir  = "D:\\SRML-1D-pulse-types\\Results\\Networks" + delim + "model" + "_" + modelname + delim + str(NEPOCHS) + "_epochs"

    savedir = "D:\\SRML-1D-pulse-types\\Results\\Figures" + delim + "model" + "_" + modelname
    
    if path.exists(savedir) == False:
        makedirs(savedir, exist_ok=True)
        
    #%% DATA SET PARAMETERS
    NDATA = 10         # Number of data files
    
    #%% LOAD THE DATASET
    ind = np.arange(0,NDATA,1)   # File indices data   
    dataset = load_dataset(datadir,filelist,ind)
        
    #%% LOAD THE TRAINED MODEL
    epoch = NEPOCHS-1
    model = DilatedCNN(hidden_size=64, depth=12)  
    modelpath = modeldir + delim + "epoch_" + str(epoch)
    model.load_state_dict(torch.load(modelpath))
        
    model = model.cuda()
    model.eval()
    
    #%% COMPUTE THE PREDICTION
    
    idx = 2     # File index
    
    V = dataset[idx]['x'].cuda().unsqueeze(0)       # The RF signal
    y = dataset[idx]['y1'].cuda().unsqueeze(0)      # The ground truth
    
    z = model(V)                                    # Prediction
    
    # Convert signal, ground truth, and prediction to numpy arrays:     
    z = torch.squeeze(z)
    y = torch.squeeze(y)
    z = z.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    V  = torch.squeeze(V)
    V  = V.detach().cpu().numpy()
    
    #%% PLOT THE RESULTS
    figWidth    = 350/25.4
    figHeight   = 150/25.4 
    lineWidth   = 0.7
    fontSize    = 20
    labelSize   = 16
    
    # Plot the RF signal:
    plt.figure(figsize=(figWidth,figHeight), dpi=150)
    plt.plot(V,color=(0.000,0.000,0.475))
    plt.xlim([1, 8446])
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.savefig(savedir + delim + modelname + '_output_RF.svg')
    
    ##############################################################################
    # Plot the ground truth and prediction (zoomed)
    
    Nstart = 3900
    Nend = 4500   
    
    fig = plt.figure(figsize = (figWidth,figHeight), dpi = 150)
 
    # Plot the ground thruth 
    plt.scatter(np.arange(0,len(y))[y>0],y[y>0],marker="d",color='r')
    plt.plot(np.arange(Nstart,Nend), z[Nstart:Nend],color=model_properties['color'])
    plt.grid()
    plt.xlabel('grid point',fontsize=fontSize,family='arial',fontweight='bold')
    plt.xlim([Nstart,Nend])
    plt.ylim([-1.1,1.1])
    plt.legend(['ground truth',model_properties['abbreviation']], fontsize = fontSize)
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=labelSize)
    
    # Add an axis for time:
    Fs = 62.5       # Sampling rate (MHz)
    ax1 = fig.axes[0]
    ax2 = ax1.twiny()
    ax2.set_xlim([Nstart/Fs,Nend/Fs])
    ax2.set_xlabel('time (\u03bcs)',fontsize=fontSize,family='arial')
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=labelSize)
    
    plt.savefig(savedir + delim + 'model_' + modelname + "_" + str(NEPOCHS) + "_epochs" + '_output_zoomed.svg')
    plt.savefig(savedir + delim + 'model_' + modelname + "_" + str(NEPOCHS) + "_epochs" + '_output_zoomed.png')
    
    ##############################################################################
    # Plot the ground truth and prediction (full scale)
    fig = plt.figure(figsize=(3.6,2.0), dpi=150)
    plt.plot(y,color=(0.122,0.467,0.706))
    plt.plot(-z,color=(1.000,0.322,0.322))
    
    plt.title('model ' + modelname + ', epoch 1249',fontsize=8,family='arial')
    plt.xlabel('grid point',fontsize=8,family='arial',fontweight='bold')
    plt.ylabel('bubble count',fontsize=8,family='arial',fontweight='bold')
    plt.xlim([1, 8446])
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Plot boundaries zoomed interval:
    plt.plot([Nstart, Nstart],[-1.2, 3.2])
    plt.plot([Nend, Nend],[-1.2, 3.2])
    plt.ylim([-1.2,3.2])
    
    # Add an axis for time:
    ax1 = fig.axes[0]
    ax2 = ax1.twiny()
    ax2.set_xlim([0,8446/Fs])
    ax2.set_xlabel('time (\u03bcs)',fontsize=8,family='arial')
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    plt.savefig(savedir + delim + 'model_' + modelname + "_" + str(NEPOCHS) + "_epochs" + "_output_full.svg")
    plt.savefig(savedir + delim + 'model_' + modelname + "_" + str(NEPOCHS) + "_epochs" + "_output_full.png")
    
    #%%
    print('number of bubbles: ' + str(sum(y)))

