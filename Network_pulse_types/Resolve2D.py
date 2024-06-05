# -*- coding: utf-8 -*-

# This code applies trained model HS_001 to 2D  RF data. This code applies the
# super-resolution neural network per batch of RF lines. All RF lines are
# concatenated again into one matrix per file.
#
# The results are stored in the original data folder

# Import packages:
import torch
import os
import numpy as np

from bubbledataloadermatlab import load_dataset_rf
from bubblenetwork import DilatedCNN
from customModelInfo import model_info
from addNoise import add_noise, V_ref, filt_a, filt_b

NEPOCHS = 1250
delim = '\\'

additional_specification = '_noise128' # Additional model specifications

#%% FILE DIRECTORIES
# Directory where the model parameters are stored:
modeldir = r'D:\SRML-1D-pulse-types\Results\Networks' 

# Directory where the simulated RF data is stored: 
filedir = r'D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D'
#filename = 'RFDATA2D00001.mat'

# Directory where the super-resolved RF data will be stored:
destdir = filedir

#%% MODELS
datasets = os.listdir(filedir)

if 'noise' in additional_specification: # Evaluate noise
    datasets = [dataset for dataset in datasets if 'pulseSingle_Reference_OneCycle_500' in dataset or 'pulseChirp_Long_Downsweep_500' in dataset or 'pulseChirp_Short_Downsweep_500' in dataset or 'pulseSingle_Long_MedF_500' in dataset or 'pulseSingle_Short_MedF_500' in dataset]

for datafolder in datasets:
    print(datafolder)
    filename = 'RFDATA00001.mat'
    
    #%% LOAD THE DATA
    BATCH_SIZE = 16     # Batch size
    
    filepath = os.path.join(filedir,datafolder,filename)
    
    dataset = load_dataset_rf(filepath) 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = BATCH_SIZE, shuffle=False)
    
    #%% LOAD THE MODEL
    pulse = datafolder[:datafolder.rfind('_')]
    
    epoch = NEPOCHS-1
    model = DilatedCNN(hidden_size=64, depth=12)  
    modelfile = 'epoch_' + str(epoch)
    modelpath = os.path.join(modeldir,'model_' + pulse + additional_specification,str(NEPOCHS)+'_epochs',modelfile)
    print('loaded model: ' + modelpath)
    
    model.load_state_dict(torch.load(modelpath))
    model = model.cuda()
    model.eval()
        
    for it, sample_batched in enumerate(dataloader):
        if 'noise' in additional_specification:
            print('noise added')
            # Determine the noise level
            noiselevel_p = int(additional_specification.split("_noise")[1])
            noiselevel = noiselevel_p*V_ref/100
            
            V   = sample_batched['x'].cpu().numpy()    # RF signals
            
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
        else:
            V = sample_batched['x'].cuda()      # RF data
       
        # Apply the model:
        z = model(V)                        # Predicted bubble distribution
               
        # Convert to numpy array:  
        z = np.squeeze(np.transpose(z.cpu().detach().numpy()))
        
        print(z.shape)
        
        # Store the results in a large matrix:
        if it == 0:
            Z = z
        else:
            Z = np.concatenate((Z,z),axis=1)
            
    print(Z.shape)
    
    
    print('Saving ' + filename)
    # Write the result to a text file:
    destfilenametxt = os.path.join(destdir,datafolder,filename[0:-4] + '_sr' + additional_specification + '.txt')
    
    with open(destfilenametxt,'w') as f:
        np.savetxt(f,Z,'%.10f',delimiter=',')
    
    destfilenamenpy = os.path.join(destdir,datafolder,filename[0:-4] + '_sr' + additional_specification + '.npy')
    np.save(destfilenamenpy,Z)

    

    
    

    
