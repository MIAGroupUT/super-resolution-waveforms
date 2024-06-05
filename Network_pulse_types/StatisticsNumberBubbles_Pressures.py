# For a given model, and for given threshold values, compute the F1 score as
# a function as number of bubbles, and as a function of the acoustic pressure.

# Import packages:
import torch
from os import listdir, path, makedirs
import numpy as np
import csv

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats
from addNoise import add_noise, V_ref, filt_a, filt_b

delim = "\\"

NEPOCHS = 1250

# Get the networks to be evaluated
network_dir = "D:\\SRML-1D-pulse-types\\Results\\Networks"
networks = listdir(network_dir)

tol_list = [1,4]

for k,modelname in enumerate(networks):
    print(modelname)
    
    if "noise" in modelname:
        noise = ''.join(filter(str.isdigit, modelname))
        noiselevel_p = int(noise)
        noiselevel = noiselevel_p*V_ref/100
        
        # Get the right model name for the raw data
        parent_model = modelname.replace('model_','')
        parent_model = parent_model.replace(noise,'')
        parent_model = parent_model.replace('_noise','')
        print("Noise level: " + str(noiselevel_p))
        continue
    elif "linear" in modelname:
        print("Simulation descriptives not present as they were not stored. Please rerun the simulations and store the simulation settings.")
        continue
    else:
        parent_model = modelname.replace('model_','')
    
    #%% FILE DIRECTORIES
    datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + parent_model + "\\TESTING"
    
    # Investigate this model:
    modeldir    = network_dir + delim + modelname  + delim + str(NEPOCHS) + '_epochs'

    # Store the results in this directory:
    savedir = modeldir + delim + 'StatisticsEvaluation'
    
    if path.exists(savedir) == False:
        makedirs(savedir, exist_ok=True)
    
    
    #%% DATA SET PARAMETERS
    NDATA = 960         # Number of data files
    BATCH_SIZE = 8
    
    #%% LOAD THE DATASET
    ind = np.arange(0,NDATA,1)   # File indices validation data
    
    filelist = listdir(datadir)
    dataset = load_dataset(datadir,filelist,ind)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    #%% LOAD THE MODEL
    torch.cuda.empty_cache()
    
    model = DilatedCNN(hidden_size=64, depth=12)  
    modelpath = modeldir + delim + 'epoch_' + str(NEPOCHS-1)
    model.load_state_dict(torch.load(modelpath)) 
    model = model.cuda()
    model.eval()
    
    #%% LOAD THE TRESHOLDS
    # Get optimal threshold values and the corresponding tolerance list:
    th_opt    = np.load(modeldir + delim + 'thresholds_optimal.npy')
    
    #%% LOAD THE SIMULATION DESCRIPTIVES
    
    with open(datadir + delim + 'simulationDescriptives.txt','r') as f:
        reader = csv.reader(f)
        descr_list = list(reader)
    
    sim_descriptives = np.array(descr_list[2:])
    NB_matrix = sim_descriptives[ind,0].astype('float64')
    PA_matrix = sim_descriptives[ind,1].astype('float64')
    
    #%% COMPUTE THE STATISTICS
    F1_matrix = np.zeros((len(tol_list),len(dataset)))  # F1 score
    
    for i, tolerance in enumerate(tol_list): 
        print('tolerance %d' % tolerance) 
        
        threshold = th_opt[tolerance] # Get the optimal threshold corresponding to the tolerance
           
        F1        = np.array([])
        
        for it, sample_batched in enumerate(data_loader):
    
            if "noise" in modelname:
                V   = sample_batched['x'].cpu().numpy()    # RF signals
                # Add noise to the RF signals and convert to torch cuda tensor:
                V = add_noise(V,noiselevel,filt_b,filt_a)
            else:
                V   = sample_batched['x'].cuda()    # RF signals
            
            y = sample_batched['y1'].cuda()     # Ground truth
            z = model(V)                        # Prediction
            
            # Get the number of bubbles:
            NB = np.sum(y.cpu().numpy(),axis=1)
                   
            # Compute statistics on the batch:
            TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
            
            R[R==0] = 1e-3     # Prevent divide by 0
    
            # Collect statistics from each prediction in a list:
            F1        = np.append(F1,2*P*R/(P+R))
    
        F1_matrix[i,:] = F1
    
    np.save(savedir + '/F1_matrix',F1_matrix)   
    np.save(savedir + '/num_bub_matrix',NB_matrix)
    np.save(savedir + '/pressure_matrix',PA_matrix)
