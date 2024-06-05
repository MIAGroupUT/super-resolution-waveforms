# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, makedirs
import torch

from bubblenetwork import DilatedCNN
from bubblenetwork import RandomModel
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats
from customModelInfo import model_info
from addNoise import add_noise, V_ref, filt_a, filt_b

delim = "\\"

includeNoise = True
includeCompressed = False
includeLinear = False

fSize = 'small'                 # small, large or combined
dispMode = 'modelComparison'    # Enter which figure you would like to make. Enter 'modelComparison', 'modelLinearization'
model_filter = 'pulse'          # Enter filter which pulse names need to be involved. For all pulses, enter 'pulse'
boxPlotMode = False

# Figure settings
if fSize == 'small':
    figWidth    = 3.5
    figHeight   = 2
else:
    figWidth    = 7.16
    figHeight   = 2.5
    
if boxPlotMode == True:
    relFigSize = 2/3
else:
    relFigSize = 1


lineWidth   = 1
fontSize    = 8
labelSize   = 8
dpi = 600
fontFamily  = "Times New Roman" 

# Model characteristics
NEPOCHS = 1250 

if dispMode == 'modelComparison':
    model_list = np.array(list(model_info.keys()))
    
    # Filter out models
    idx_filter = [model_list[k].__contains__(model_filter) for k,model in enumerate(model_list)]
    model_list = list(model_list[idx_filter])
    
elif dispMode == 'modelLinearization':
    model_list = ["pulseChirp_Short_Downsweep_compressed","pulseChirp_Short_Downsweep_linear", "pulseChirp_Short_Downsweep"]

# Get the networks to be evaluated
network_dir = "D:\\SRML-1D-pulse-types\\Results\\Networks"
networks = listdir(network_dir)

final_epochs = [1249]

#%% DATA SET PARAMETERS
NDATA = 960         # Number of data files
BATCH_SIZE = 8      # Batch size

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = fontFamily

for m,final_epoch in enumerate(final_epochs):
    # Initialize the figures
    
    if m == 0:
        fig = plt.figure(figsize=(relFigSize*figWidth,figHeight),dpi=dpi)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        
        if boxPlotMode == True:
            figBox = plt.figure(figsize=((1-relFigSize)*figWidth,figHeight),dpi=dpi)
            axBox = figBox.add_subplot(111)
            labels_boxes = []
            F1_boxes = []
            colors_boxes = []
        
    for k,modelname in enumerate(model_list):
        
        print(modelname)  
        
        if "noise" in modelname:
            if includeNoise == False:
                continue
            
            noise = ''.join(filter(str.isdigit, modelname))
            noiselevel_p = int(noise)
            noiselevel = noiselevel_p*V_ref/100
            
            # Get the right model name for the raw data
            parent_model = modelname.replace('model_','')
            parent_model = parent_model.replace(noise,'')
            parent_model = parent_model.replace('_noise','')
            print("Noise level: " + str(noiselevel_p))
            
            datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + parent_model + "\\TESTING"
            
        elif "linear" in modelname:
            if includeLinear == False:
                continue
            
            parent_model = modelname.replace('model_','')
            datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + parent_model + "\\TESTING"
            
            parent_model = modelname.replace('_linear','')
        elif "compressed" in modelname:
            if includeCompressed == False:
                continue
            parent_model = modelname.replace('model_','')
            datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + parent_model + "\\TESTING"
            
        else:
            parent_model = modelname.replace('model_','')
            datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + parent_model + "\\TESTING"
        
        model_properties = model_info[parent_model]
        
        #%% FILE DIRECTORIES
        filelist    = listdir(datadir)   
        
        modeldir    = network_dir + delim + "model_" + modelname  + delim + str(NEPOCHS) + '_epochs'
        
        savedir     = "D:\\SRML-1D-pulse-types\\Results\\Figures"
        
        if path.exists(savedir) == False:
            makedirs(savedir, exist_ok=True)
            
        #%% LOAD THE DATASET
        ind = np.arange(0,NDATA,1)   # File indices validation data
            
        dataset = load_dataset(datadir,filelist,ind)
    
        data_loader = torch.utils.data.DataLoader(
            dataset,   batch_size=BATCH_SIZE, shuffle=False)
        
        #%% FOR EACH MODEL AND TOLERANCE COMPUTE THE F1 SCORE
        th_opt    = np.load(modeldir + delim + "thresholds_optimal.npy")
        tol_list  = np.load(modeldir + delim + "tolerance_list.npy")
        
        # LOAD THE MODEL:
            
        torch.cuda.empty_cache()
        
        if modelname == 'R':
            model = RandomModel()
        
        else:   
            model = DilatedCNN(hidden_size=64, depth=12)  
            modelpath = modeldir + delim + 'epoch_' + str(final_epoch)
            model.load_state_dict(torch.load(modelpath))
            
        model = model.cuda()
        model.eval()
        
        # List with F1 scores for each tolerance:
        F1_array = np.zeros(len(tol_list))
      
        for i, tolerance in enumerate(tol_list): 
            print('tolerance %d' % tolerance) 
    
            threshold = th_opt[i]   	# Optimal threshold for this tolerance
            F1        = np.array([])    # List with F1 scores for each prediction
            
            for it, sample_batched in enumerate(data_loader):
                
                if "noise" in modelname:
                    V   = sample_batched['x'].cpu().numpy()    # RF signals
                    # Add noise to the RF signals and convert to torch cuda tensor:
                    V = add_noise(V,noiselevel,filt_b,filt_a)
                else:
                    V   = sample_batched['x'].cuda()    # RF signals
                    
                y = sample_batched['y1'].cuda()     # Ground truth
                z = model(V)                        # Prediction
                
                # Compute statistics on the batch:
                TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
                
                P[P==0] = 1e-3     # Prevent divide by 0
                R[R==0] = 1e-3     # Prevent divide by 0

    
                # Collect F1 score each prediction in a list:
                F1        = np.append(F1,2*P*R/(P+R))
                
            # Plot the boxplot at tolerance = 4    
            if tolerance == 4 and boxPlotMode == True:
                F1_boxes.append(F1)
                labels_boxes.append(model_properties['abbreviation'])
                colors_boxes.append(model_properties['color'])
                
                figHist, axHist = plt.subplots()
                axHist.hist(F1)
                axHist.set_title(modelname)
                axHist.set_ylabel('Counts')
                axHist.set_xlabel('F1 score')
                
            F1_array[i] = np.mean(F1)
        np.save(modeldir + delim + "F1_test_" + modelname + '_' + str(final_epoch+1) +'_epochs',F1_array) 
        
        if "compressed" in modelname:
            LineStyle = (0, (3, 1, 1, 1, 1, 1))
            abbreviation = model_properties['abbreviation']
        elif "linear" in modelname:
            LineStyle = 'solid'
            abbreviation = model_properties['abbreviation'] + "l"
        else:
            LineStyle = model_properties['linestyle']
            abbreviation = model_properties['abbreviation']
            
        ax1.plot(F1_array,color = model_properties['color'], linestyle = LineStyle, linewidth = lineWidth, label = abbreviation)
    
    
    ax1.set_xlabel('tolerance (grid points)',fontsize=fontSize)
    ax1.set_ylabel('F1 score',fontsize=fontSize)
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,9])
    ax1.set_xticks(np.arange(0,10))
    ax1.tick_params(labelsize=labelSize)
    ax1.grid()
    
    # Make the legend
    lgd = ax1.legend(fontsize=fontSize, loc='right')
    
    # Set the ticks and labels for the upper x-axis (ax2)
    ax2.grid()
    ax2.set_xlabel('tolerance (microm)', fontsize = fontSize)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    new_tick_labels = [f'{tick:.1f}' for tick in ax1.get_xticks()*(1480*16e-9)/1e-6]
    ax2.set_xticklabels(new_tick_labels, rotation=45)
    ax2.tick_params(labelsize=labelSize)
    fig.savefig(savedir + delim + dispMode + "_" + str(final_epoch+1) + ".svg")
    
    if boxPlotMode == True: # Make the boxplot
        
        bplot = axBox.boxplot(F1_boxes, whis=[5,95], sym='', showmeans=True, vert=True, patch_artist = True, labels=labels_boxes, meanprops=dict(marker='o',markersize=1, markerfacecolor='k'))
        axBox.tick_params(axis='x', labelrotation=-45, labelsize=fontSize)
        axBox.tick_params(axis='y', labelsize=fontSize)
        axBox.set_title('At tol = 4')
        axBox.set_ylabel('F1 score')
        axBox.set_ylim([0.6,1])
        axBox.grid()
        for patch, color in zip(bplot['boxes'],colors_boxes):
            patch.set_facecolor(color)
        figBox.savefig(savedir + delim + "ModelComparison_boxplot_" + str(final_epoch+1) + ".svg")
