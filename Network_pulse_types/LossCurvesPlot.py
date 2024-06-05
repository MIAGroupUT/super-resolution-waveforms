# -*- coding: utf-8 -*-

# In this file, the total loss is displayed as a representation of the training 
# progression for each pulse network

import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs
from customModelInfo import model_info

#%% INITIALIZE
delim = "\\"
includeLinear = False
includeCompressed = True

parentdir = "D:\\SRML-1D-pulse-types\\Results\\Networks"
final_epochs = [1249]

# Initialize figure and plot parameters
figWidth    = 3.5
figHeight   = 2.5
lineWidth   = 1
fontSize    = 8
labelSize   = 8

# Initialize directories
model_list = list(model_info.keys())
modeldirs = [(parentdir + delim + 'model_' + modelname) for modelname in model_list]

savedir = "D:\\SRML-1D-pulse-types\\Results\\Figures"


# Make a new directory if the savedir does not exist
if path.exists(savedir) == False:
    makedirs(savedir, exist_ok=True)

for m, final_epoch in enumerate(final_epochs):
        
    #%% DATASTRUCTURES
    
    final_trn_losses = np.zeros(len(modeldirs))
    final_val_losses = np.zeros(len(modeldirs))
    datasetsizes = [1024] 
    
    # Initialize the figure
    fig = plt.figure(figsize=(figWidth,figHeight), dpi=300)
    ax = plt.axes()
    epochs = np.linspace(0,final_epoch, num=final_epoch+1, dtype='int64')
    
    # For each dataset size, plot the evolution of the validation loss, and 
    # collect the final training and validation loss.
    for k, modeldir in enumerate(modeldirs):
        modelname = model_list[k]
        
        if "linear" in modelname:
            if includeLinear == False:
                continue
        
        elif "compressed" in modelname:
            if includeCompressed == False:
                continue
            
        model_properties = model_info[modelname]
        
        #%% Load the data
        val_loss  = np.load(modeldir + delim + str(final_epoch+1) +'_epochs' + delim + 'val_loss.npy')
        trn_loss  = np.load(modeldir + delim + str(final_epoch+1) +'_epochs' + delim + 'train_loss.npy')
        
        final_val_losses[k] = val_loss[-1]
        final_trn_losses[k] = trn_loss[-1]
        
        #%% Plot the results
        ax.plot(epochs,val_loss,color=model_properties['color'], linewidth=lineWidth, label=model_properties['abbreviation'])
    
    # Format the figure
    ax.axis([0,final_epoch,0,2])
    ax.grid(which='both')
    plt.yscale('log')    
    ax.set_ylim([1, 5])
    ax.tick_params(axis='both', which='major', labelsize=labelSize)
    ax.tick_params(axis='y', which='both', labelsize=labelSize)
    ax.legend(title="pulse type:",fontsize=fontSize,title_fontsize=fontSize, loc='center left', bbox_to_anchor = (0.85,0.5))
    plt.xlabel('epochs',fontsize=fontSize)
    plt.ylabel('Total loss',fontsize=fontSize)
    plt.title('Validation loss as a function of training epochs',fontsize=fontSize)
    
    # Save the figure as an .svg file
    plt.savefig(savedir + delim + 'TrainingProgression.svg')
