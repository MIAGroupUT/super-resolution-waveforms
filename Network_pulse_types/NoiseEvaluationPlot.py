# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, makedirs
import torch
from scipy import signal

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats
from customModelInfo import model_info
from addNoise import add_noise, noiselevels_p, noiselevels, filt_b, filt_a

delim = '//'

NEPOCHS = 1250

NDATA = 960
BATCH_SIZE = 8      # Batch size

# Models evaluated in noise
model_list = ["pulseSingle_Reference_OneCycle","pulseChirp_Long_Downsweep","pulseChirp_Short_Downsweep","pulseSingle_Short_MedF","pulseSingle_Long_MedF"]

#%% PLOT SETTINGS
figWidth    = 3.5
figHeight   = 2#figWidth*(3/4)
lineWidth   = 1
fontSize    = 8
labelSize   = 8
fontFamily  = "Times New Roman" 
dpi = 600

tol_list = [1,4]


#%% NOISE EVALUATION
for i,tol in enumerate(tol_list):
    plt.rcParams['font.family'] = fontFamily
    for k,modelname in enumerate(model_list):
        
        # Load model properties
        model_properties = model_info[modelname]
        
        # List with F1 scores for each noise level:
        F1_array = np.zeros(len(noiselevels_p))
        
        for j,noiselevel in enumerate(noiselevels):
            print(modelname, noiselevels_p[j])
            
            #%% FILE DIRECTORIES
            datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + modelname + "\\TESTING"
            filelist    = listdir(datadir)   
            
            if noiselevel == 0:
                modeldir    = "D:\\SRML-1D-pulse-types\\Results\\Networks\\model_" + modelname + delim + str(NEPOCHS) + "_epochs"
            else:
                modeldir    = "D:\\SRML-1D-pulse-types\\Results\\Networks\\model_" + modelname + "_noise" + str(noiselevels_p[j]) + delim + str(NEPOCHS) + "_epochs"
            
            savedir     = "D:\\SRML-1D-pulse-types\\Results\\Figures"
            
            #%% LOAD THE DATASET
            ind = np.arange(0,NDATA,1)   # File indices validation data
                
            dataset = load_dataset(datadir,filelist,ind)
    
            data_loader = torch.utils.data.DataLoader(
                dataset,   batch_size=BATCH_SIZE, shuffle=False)
            #%% FOR EACH MODEL AND TOLERANCE COMPUTE THE F1 SCORE
            
            th_opt    = np.load(modeldir + delim + "thresholds_optimal.npy")
            
            # LOAD THE MODEL:
                
            torch.cuda.empty_cache()
            epoch = NEPOCHS-1
            
            model = DilatedCNN(hidden_size=64, depth=12)  
            modelpath = modeldir + delim + 'epoch_' + str(epoch)
            model.load_state_dict(torch.load(modelpath))
            
            model = model.cuda()
            model.eval()
            
            threshold = th_opt[tol] # Load the optimized threshold
            F1        = np.array([])    # List with F1 scores for each prediction
            
            for it, sample_batched in enumerate(data_loader):
                
                V = sample_batched['x'].cpu().numpy()    # RF signals
            
                # Add noise to the RF signals and convert to torch cuda tensor:
                V = add_noise(V,noiselevel,filt_b,filt_a)
                y = sample_batched['y1'].cuda()     # Ground truth
                z = model(V)                        # Prediction
                
                # Compute statistics on the batch:
                TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tol,threshold) 
                
                P[P==0] = 1e-3     # Prevent divide by 0
                R[R==0] = 1e-3     # Prevent divide by 0
    
                # Collect F1 score each prediction in a list:
                F1        = np.append(F1,2*P*R/(P+R))
            
            # Store the results
            np.save(modeldir + delim + 'Noise_F1',F1)
            
            F1_array[j] = np.mean(F1)
        
        
        #%% PLOT THE RESULTS
        if k == 0: #Initialize the figure
            fig,ax = plt.subplots(1,1, figsize=(figWidth,figHeight), dpi = dpi)
            ax.set_xscale('log')
    
        ax.plot(noiselevels_p, F1_array, color = model_properties['color'], linestyle = model_properties['linestyle'], linewidth = lineWidth, label = model_properties['abbreviation'])
    
    # Format and save the figure
    ax.set_xlabel('Noise level (% of Vref)', fontsize = fontSize, family=fontFamily)
    ax.set_ylabel('F1 score', fontsize = fontSize, family=fontFamily)
    ax.set_ylim([0,1])
    ax.set_title("Model performance versus noise level (tolerance = {} grid points)".format(tol), fontsize = fontSize, family=fontFamily)
    
    # Ticks
    ax.tick_params(labelsize=labelSize)
    ax.set_xticks(noiselevels_p)
    ax.set_xticklabels(noiselevels_p, family=fontFamily)
    plt.minorticks_off()
    ax.xaxis.grid()
    ax.yaxis.grid()
    
    # Make the legend
    lgd = ax.legend(fontsize=fontSize, loc='right')
    figName = "NoiseEvaluation_%d.svg" % (tol)
    plt.savefig(savedir + delim + figName)
        
        
        
        
        
        