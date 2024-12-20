# -*- coding: utf-8 -*-

# This code applies a trained model to 2D  RF data. This code applies the
# super-resolution neural network per batch of RF lines. All RF lines are
# concatenated again into one matrix per file.
#
# The results are stored in the original data folder

# Import packages:
import torch
import os
import numpy as np
import tkinter as tk

from bubbledataloadermatlab import load_dataset_rf
from bubblenetwork import DilatedCNN
from customModelInfo import model_info
from addNoise import add_noise, V_ref, filt_a, filt_b
from tkinter import filedialog

NEPOCHS = 1250
delim = '\\'

#%% INPUTS
# Additional model specifications, enter '_noise#' with # representing the number, 'noiserange#1-#2 with #1 and #2 representing the noise levels in the range or '_absolutenoise#' with # representing an absolute noise voltage
additional_specification = '_noise128'
# additional_specification = '_noiserange4-7' 
# additional_specification = '_absolutenoise0.752713' 
#additional_specification = ''

# Flag if the data is experimental
experimental_data = False

# Flag to process all data
process_all = True

tolerance = 4 # Tolerance applied to the model. Enter any integer in the range [0,9], or None if you dont want to apply a threshold. The script selects the corresponding threshold, and deletes all predictions below this value.

#%% FILE DIRECTORIES
# Directory where the model parameters are stored:
modeldir = r'D:\SRML-1D-pulse-types\Results\Networks' 

# Directory where the  RF data is stored: 

def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select the folder containing the processed 2D mat files")
    folder_path = folder_path.replace("/", delim)
    return folder_path

filedir = select_directory()
if filedir:
    print(f"Selected folder: {filedir}")
else:
    print("No folder selected.")

# If the code above does not prompt a file, you can alternatively set the directory manually
filedir = r"D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D"

# Directory where the super-resolved RF data will be stored:
destdir = filedir

# Filter the datasets
datasets = os.listdir(filedir)
if 'noiserange' in additional_specification  or 'absolutenoise' in additional_specification:
    pass
elif 'noise' in additional_specification: # Evaluate noise
    datasets = [dataset for dataset in datasets if 'pulseSingle_Reference_OneCycle_500' in dataset or 'pulseChirp_Long_Downsweep_500' in dataset or 'pulseChirp_Short_Downsweep_500' in dataset or 'pulseSingle_Long_MedF_500' in dataset or 'pulseSingle_Short_MedF_500' in dataset]

#%% RESOLVE THE DATA

for datafolder in datasets:
    print(datafolder)
    if experimental_data == False:
        if 'pulseExp' in datafolder:
            continue

    folderfiles = os.listdir(filedir + delim + datafolder)
    
    if process_all == False:
        filenames = [[file for file in folderfiles if 'RFDATA' in file][0]]
    elif process_all == True:
        filenames = [file for file in folderfiles if 'RFDATA' in file]
        filenames = [file for file in filenames if '.mat' in file]
    
    for filename in filenames:
        
        #%% LOAD THE DATA
        BATCH_SIZE = 16     # Batch size
        
        filepath = os.path.join(filedir,datafolder,filename)
        print(filepath)

        dataset = load_dataset_rf(filepath) 
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size = BATCH_SIZE, shuffle=False)
        
        #%% LOAD THE MODEL
        if 'noiserange' in additional_specification:
            pulse = datafolder
        elif 'absolutenoise' in additional_specification:
            pulse = datafolder
        elif 'noise' in additional_specification:
            pulse = datafolder[:datafolder.rfind('_')]
        else:
            pulse = datafolder

        if 'bubbles' in pulse:
            parts = pulse.split('_')
            pulse = "_".join(parts[0:len(parts)-1])

        # Load the model
        epoch = NEPOCHS-1
        model = DilatedCNN(hidden_size=64, depth=12)  
        modelfile = 'epoch_' + str(epoch)
        modelpath = os.path.join(modeldir,'model_' + pulse + additional_specification,str(NEPOCHS)+'_epochs',modelfile)
        print('loaded model: ' + modelpath)
        
        model.load_state_dict(torch.load(modelpath))
        model = model.cuda()
        model.eval()
        
        #%% LOAD THE THRESHOLDS
        if tolerance is not None:
            th_opt    = np.load(modeldir + delim + 'model_' + pulse + additional_specification + delim + str(NEPOCHS) + '_epochs' + delim + "thresholds_optimal.npy")
            tol_list  = np.load(modeldir + delim + 'model_' + pulse + additional_specification + delim + str(NEPOCHS) + '_epochs' + delim + "tolerance_list.npy")

            th = th_opt[tol_list == tolerance]
            
        for it, sample_batched in enumerate(dataloader):
            if experimental_data == True:
                V = sample_batched['x'].cuda()      # RF data

            elif 'noise' in additional_specification:
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
            
            # Remove values below threshold
            if tolerance is not None:
                z[z<th] = 0
            
            print(z.shape)
            
            # Store the results in a large matrix:
            if it == 0:
                Z = z
            else:
                Z = np.concatenate((Z,z),axis=1)
                
        print(Z.shape)
        
        
        savedir = os.path.join(destdir,datafolder,'sr_data')
        
        if os.path.exists(savedir)==False:
            os.mkdir(savedir)
        
        print('Saving ' + filename)
        if tolerance is None:
            newFilename = os.path.join(savedir,filename[0:-4] + '_sr' + additional_specification)
        else:
            newFilename = os.path.join(savedir,filename[0:-4] + '_sr' + '_tol' + str(tolerance) + additional_specification)
        
        # Write the result to a text file:
        destfilenametxt = newFilename + '.txt'
        
        with open(destfilenametxt,'w') as f:
            np.savetxt(f,Z,'%.10f',delimiter=',')
        
        destfilenamenpy = newFilename + '.npy'
        np.save(destfilenamenpy,Z)

    

    
    

    
