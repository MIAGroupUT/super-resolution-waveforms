"""
Script to optimize the threshold for every tolerance.

It loops through a series of models, and computes the optimal threshold for each threshold in the range [0, 9] grid points.

Author: Nathan Blanken and Rienk Zorgdrager, University of Twente, 2024 
"""

# Import packages:
import torch
from os import listdir, path, makedirs
import numpy as np
import platform
import random
import re

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats
from customModelInfo import model_info
from addNoise import add_noise, V_ref, filt_a, filt_b


delim = "\\"

NEPOCHS = 1250

#%% Get a list of models of which the thresholds need to be optimized

# Filter out the models of interest. For all models, uncomment:
# model_list = np.array(list(model_info.keys()))

# To compute only the models without noise, uncomment:
# model_list = [model for model in model_list if "noise" not in model]

# To compute only the models trained on a noise level from 4 to 256% of V_ref, uncomment:
# model_list = [model for model in model_list if "noiserange4-256"  in model]

# To compute only the models analyzed in the Results section, uncomment:
# model_list      = ["pulseSingle_Reference_OneCycle","pulseChirp_Long_Downsweep","pulseChirp_Short_Downsweep","pulseSingle_Short_MedF","pulseSingle_Long_MedF"]

# To compute only the models used for experimental validation, uncomment:
model_list = ["pulseExpVal_Short_SIP", "pulseExpVal_Short_chirp"]

# Additional filtering of specific networks. 
filt = "4_absolutenoise0.752713" # Use only networks with a tolerance of 4 grid points and an absolute noise value of 0.752713
model_list = [model + filt for model in model_list]

# homedir = "/home/zorgdragerrc/SRML-1D-pulse-types/Results"
homedir = "D:\SRML-1D-pulse-types\Results"

# Get the networks to be evaluated
network_dir = homedir + delim + "Networks"
networks = listdir(network_dir)
networks = [network for network in networks if filt in network]

#%% Loop through the networks
for k,modelname in enumerate(networks):
    print(modelname)

    if "_noise" in modelname:
        noise = ''.join(filter(str.isdigit, modelname))
        if "_noiserange" in filt:
            parts = filt.split('-')
            
            noise1 = int(parts[0].split('noiserange')[-1])
            noise2 = int(parts[1])
        noiselevel_p = int(noise)
        noiselevel = noiselevel_p*V_ref/100
            
        # Get the right model name for the raw data
        parent_model = modelname.replace('model_','')
        parent_model = parent_model.replace(noise,'')
        parent_model = parent_model.replace(filt,'')
        #print("Noise level: " + str(noiselevel_p))
    elif "_absolutenoise" in modelname:
        noise = re.findall(r"\d+\.\d+", modelname)
        noiselevel = float(noise[0])
        
        # Get the right model name for the raw data
        parent_model = modelname.replace('model_','')
        parent_model = parent_model.replace(filt[1:],'')        
        
    else:
        parent_model = modelname.replace('model_','')
        
    model_properties = model_info[parent_model]

    #%% FILE DIRECTORIES
    datadir = homedir + delim + "RF signals" + delim + "txt_files" + delim + parent_model + delim + "VALIDATION"
    filelist = listdir(datadir)
    
    # Investigate this model:
    modeldir = network_dir + delim + modelname  + delim + str(NEPOCHS) + '_epochs'
    
    # Store the results in this directory:
    savedir = modeldir
    
    if path.exists(savedir) == False:
        makedirs(savedir, exist_ok=True)
        
    #%% DATA SET PARAMETERS
    NDATA = 960         # Number of data files
    BATCH_SIZE = 64     # Batch size
    
    #%% LOAD THE DATASET
    ind = np.arange(0,NDATA,1)   # File indices validation data
        
    dataset = load_dataset(datadir,filelist,ind)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    #%% LOAD THE MODEL
    epoch = NEPOCHS-1

    model = DilatedCNN(hidden_size=64, depth=12)  
    modelfile = 'epoch_' + str(epoch)
    modelpath = modeldir + delim + modelfile
    model.load_state_dict(torch.load(modelpath))
        
    model = model.cuda()
    model.eval()
    
    #%% COMPUTE THE F1 scores
    th_list1  = np.arange(-5.0,-1.0,0.5)
    th_list1  = np.power(10,th_list1)
    th_list2  = np.arange(0.1,0.95,0.05)
    
    # List of thresholds
    th_list = np.concatenate((th_list1, th_list2))  
    
    # List of tolerances     
    tol_list = np.arange(0,10)
    
    # Matrix with F1 scores for each tolerance and each threshold:
    F1_matrix = np.zeros((len(tol_list),len(th_list)))
    
    # For each tolerance and threshold, compute the average F1 score on 
    # the dataset:
      
    for i, tolerance in enumerate(tol_list): 
        print('tolerance %d' % tolerance) 
    
        for j,threshold in enumerate(th_list):
            
            F1        = np.array([])
            
            for it, sample_batched in enumerate(data_loader):
                
                if "_noiserange" in filt:
                    noiselevel = np.float64(random.uniform(noise1, noise2))*V_ref / 100

                if "noise" in modelname:
                    V   = sample_batched['x'].cpu().numpy()    # RF signals
                    # Add noise to the RF signals and convert to torch cuda tensor:
                    V   = add_noise(V,noiselevel,filt_b,filt_a)
                else:
                    V   = sample_batched['x'].cuda()    # RF signals
                    
                y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
                z = model(V)                        # Predicted bubble distribution
                
                # Compute statistics on the batch:
                TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
                
                P[P==0] = 1e-3     # Prevent divide by 0
                R[R==0] = 1e-3     # Prevent divide by 0
    
                # Collect F1 score each prediction in a list:
                F1        = np.append(F1,2*P*R/(P+R))
    
            F1_matrix[i,j] = np.mean(F1)
    
    # List with optimal thresholds for each tolerance:
    th_opt = th_list[np.argmax(F1_matrix,1)]
    
    # Save the results
    np.save(savedir + '/thresholds_optimal',    th_opt)
    np.save(savedir + '/tolerance_list',        tol_list)