# -*- coding: utf-8 -*-
"""
Trains the parsed pulse.

Author: Rienk Zorgdrager, University of Twente, 2024
"""

#%% LOAD THE PACKAGES
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os import listdir
from os import path, makedirs
import argparse
import random

from bubblenetwork import DilatedCNN
from bubbledataloader import load_dataset
from bubblelossfunctions import dual_loss
from bubblelogging import LogVars
from bubblelogging import BubbleLosses
from addNoise import add_noise, noiselevels_p, noiselevels, filt_b, filt_a, V_ref

def train(pulse_name):
    # Train the network on the provided pulse in a noise-free situation.

    print('Training network on {}'.format(pulse_name))
    torch.cuda.empty_cache()
    
    #%% SYSTEM SETTINGS
    delim = "\\"
    
    #%% FILE DIRECTORIES
    
    trndir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\TRAINING"
    valdir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\VALIDATION"
    
    # Exclude non-RF datafiles
    trnfilelist = listdir(trndir)
    idxRF = [index for index, string in enumerate(trnfilelist) if 'RF' in string]
    trnfilelist = [trnfilelist[i] for i in idxRF]
    
    valfilelist = listdir(valdir)
    idxRF = [index for index, string in enumerate(valfilelist) if 'RF' in string]
    valfilelist = [valfilelist[i] for i in idxRF]
    
    #%% TRAINING PARAMETERS
    NEPOCHS = 1250      # Number of epochs
    NTRN = 1024         # Number of training samples
    NVAL = 960          # Number of validation samples
    BATCH_SIZE = 64     # Batch size
    
    epsilon1 = 1        # Proportionality constant soft loss
    epsilon2 = 1.6      # Proportionality constant Dice loss
    a = 0.1             # Width paramater gaussian convolution kernel  
    
    #%% LOAD THE DATASETS
    trn_ind = np.arange(0,NTRN,1)   # File indices training data   
    val_ind = np.arange(0,NVAL,1)   # File indices validation data
        
    trn_dataset = load_dataset(trndir,trnfilelist,trn_ind)  # Training dataset
    val_dataset = load_dataset(valdir,valfilelist,val_ind)  # Validation dataset
    
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
    
    #%% PATHS
    savedir = "D:\\SRML-1D-pulse-types\\Results\\Networks" + delim + "model_" + pulse_name + delim + str(NEPOCHS) + "_epochs"
    print("Weights stored in", savedir)
    
    if path.exists(savedir) == False:
        makedirs(savedir, exist_ok=True)
        
    # Initialise the network
    model = DilatedCNN(hidden_size=64, depth=12)
    model = model.cuda()
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Reduce learning rate for last 250 epochs: 
    scheduler = StepLR(optimizer, step_size=(NEPOCHS-250), gamma=0.1)
    
    # Preallocate log arrays
    logging_variables = LogVars(NEPOCHS)        # Log object for tracking losses
    
    for epoch in range(NEPOCHS):
        
        epoch_losses = BubbleLosses()          # Log object for losses
        
        #%% TRAINING
        model.train()
        optimizer.zero_grad()
        for it, sample_batched in enumerate(trn_dataloader):
            
            V   = sample_batched['x'].cuda()    # RF signals
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            
            # Forward pass
            z = model(V)                        # Predicted bubble distribution
            
            # Compute loss (regression loss, classification loss, and total loss)    
            loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
            
            # Update training losses log object
            epoch_losses.update_trn_metrics(loss_r,loss_b,loss)
        
            # Backpropagation
            loss.backward()
          
        # Update network parameters:
        optimizer.step()
        
        #%% VALIDATION
        model.eval()
        for it, sample_batched in enumerate(val_dataloader):
            
            V   = sample_batched['x'].cuda()    # RF signals
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            
            # Forward pass
            z = model(V)                        # Predicted bubble distribution
            
            # Compute loss (regression loss, classification loss, and total loss)       
            loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
            
            # Update validation losses log object
            epoch_losses.update_val_metrics(loss_r,loss_b,loss)
        
        scheduler.step()
        
        #%% LOGGING AND SAVING
        # Divide cumulative losses by length of dataloader
        epoch_losses.normalize(len(trn_dataloader), len(val_dataloader))
        
        # Update logging arrays and print log message  
        logging_variables.update(epoch, NEPOCHS, epoch_losses)
        
        # Save model and logging arrays
        modelfile = 'epoch_' + str(epoch)
        modelpath = savedir + delim + modelfile
        torch.save(model.state_dict(), modelpath)
        logging_variables.save(savedir)
        
def train_noise(pulse_name):
    # Train the network on the given pulse with a fixed noise level

    print('Training network on {} with noise'.format(pulse_name))
    torch.cuda.empty_cache()
    
    #%% SYSTEM SETTINGS
    delim = "\\"
    
    #%% TRAINING PARAMETERS
    NEPOCHS = 1250      # Number of epochs
    NTRN = 1024         # Number of training samples
    NVAL = 960          # Number of validation samples
    BATCH_SIZE = 64     # Batch size
    
    epsilon1 = 1        # Proportionality constant soft loss
    epsilon2 = 1.6      # Proportionality constant Dice loss
    a = 0.1             # Width paramater gaussian convolution kernel  
    
    #%% FILE DIRECTORIES
        
    for j,noiselevel in enumerate(noiselevels):
        print("pulse name: {}, noise level: {}".format(pulse_name,noiselevels_p[j]))
        trndir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\TRAINING"
        valdir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\VALIDATION"
        
        # Exclude non-RF datafiles
        trnfilelist = listdir(trndir)
        idxRF = [index for index, string in enumerate(trnfilelist) if 'RF' in string]
        trnfilelist = [trnfilelist[i] for i in idxRF]
        
        valfilelist = listdir(valdir)
        idxRF = [index for index, string in enumerate(valfilelist) if 'RF' in string]
        valfilelist = [valfilelist[i] for i in idxRF]
        
        
        #%% LOAD THE DATASETS
        trn_ind = np.arange(0,NTRN,1)   # File indices training data   
        val_ind = np.arange(0,NVAL,1)   # File indices validation data
            
        trn_dataset = load_dataset(trndir,trnfilelist,trn_ind)  # Training dataset
        val_dataset = load_dataset(valdir,valfilelist,val_ind)  # Validation dataset
        
        trn_dataloader = torch.utils.data.DataLoader(
            trn_dataset, batch_size = BATCH_SIZE, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
        
        #%% PATHS
        savedir   = "D:\\SRML-1D-pulse-types\\Results\\Networks"  + delim + "model_" + pulse_name + '_noise' + str(noiselevels_p[j]) + delim + str(NEPOCHS) + "_epochs"
        
        print("Weights stored in: ", savedir)
        if path.exists(savedir) == False:
            makedirs(savedir, exist_ok=True)
            
        # Initialise the network
        model = DilatedCNN(hidden_size=64, depth=12)
        model = model.cuda()
        
        # Training settings
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Reduce learning rate for last 250 epochs: 
        scheduler = StepLR(optimizer, step_size=(NEPOCHS-250), gamma=0.1)
        
        # Preallocate log arrays
        logging_variables = LogVars(NEPOCHS)        # Log object for tracking losses
        
        for epoch in range(NEPOCHS):
               
            epoch_losses = BubbleLosses()          # Log object for losses
            
            #%% TRAINING
            model.train()
            optimizer.zero_grad()
            for it, sample_batched in enumerate(trn_dataloader):
                
                V = sample_batched['x'].cpu().numpy()    # RF signals
        
                # Add noise to the RF signals and convert to torch cuda tensor:
                V = add_noise(V,noiselevel,filt_b,filt_a)
                
                y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
                
                # Forward pass
                z = model(V)                        # Predicted bubble distribution
                
                # Compute loss (regression loss, classification loss, and total loss)    
                loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
                
                # Update training losses log object
                epoch_losses.update_trn_metrics(loss_r,loss_b,loss)
            
                # Backpropagation
                loss.backward()
              
            # Update network parameters:
            optimizer.step()
            
            #%% VALIDATION
            model.eval()
            for it, sample_batched in enumerate(val_dataloader):
                
                V = sample_batched['x'].cpu().numpy()    # RF signals
        
                # Add noise to the RF signals and convert to torch cuda tensor:
                V = add_noise(V,noiselevel,filt_b,filt_a)
                
                y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
                
                # Forward pass
                z = model(V)                        # Predicted bubble distribution
                
                # Compute loss (regression loss, classification loss, and total loss)       
                loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
                
                # Update validation losses log object
                epoch_losses.update_val_metrics(loss_r,loss_b,loss)
            
            scheduler.step()
            
            #%% LOGGING AND SAVING
            # Divide cumulative losses by length of dataloader
            epoch_losses.normalize(len(trn_dataloader), len(val_dataloader))
            
            # Update logging arrays and print log message  
            logging_variables.update(epoch, NEPOCHS, epoch_losses)
            
            # Save model and logging arrays
            modelfile = 'epoch_' + str(epoch)
            modelpath = savedir + delim + modelfile
            torch.save(model.state_dict(), modelpath)
            logging_variables.save(savedir)
            
def train_noise_range(pulse_name, low_noise_p, high_noise_p):
    # Train the network on a given noise range.

    print('Training network on {} with a range of noise from {} to {} percent'.format(pulse_name, low_noise_p, high_noise_p))
    torch.cuda.empty_cache()
    
    #%% SYSTEM SETTINGS
    delim = "\\"
    
    #%% TRAINING PARAMETERS
    NEPOCHS = 1250      # Number of epochs
    NTRN = 1024         # Number of training samples
    NVAL = 960          # Number of validation samples
    BATCH_SIZE = 64     # Batch size
    
    epsilon1 = 1        # Proportionality constant soft loss
    epsilon2 = 1.6      # Proportionality constant Dice loss
    a = 0.1             # Width paramater gaussian convolution kernel  
    
    #%% FILE DIRECTORIES

    trndir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\TRAINING"
    valdir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\VALIDATION"
    
    # Exclude non-RF datafiles
    trnfilelist = listdir(trndir)
    idxRF = [index for index, string in enumerate(trnfilelist) if 'RF' in string]
    trnfilelist = [trnfilelist[i] for i in idxRF]
    
    valfilelist = listdir(valdir)
    idxRF = [index for index, string in enumerate(valfilelist) if 'RF' in string]
    valfilelist = [valfilelist[i] for i in idxRF]
    
    
    #%% LOAD THE DATASETS
    trn_ind = np.arange(0,NTRN,1)   # File indices training data   
    val_ind = np.arange(0,NVAL,1)   # File indices validation data
        
    trn_dataset = load_dataset(trndir,trnfilelist,trn_ind)  # Training dataset
    val_dataset = load_dataset(valdir,valfilelist,val_ind)  # Validation dataset
    
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
    
    #%% PATHS
    savedir   = "D:\\SRML-1D-pulse-types\\Results\\Networks"  + delim + "model_" + pulse_name + '_noiserange' + str(low_noise_p) + "-" + str(high_noise_p) + delim + str(NEPOCHS) + "_epochs"
    
    print("Weights stored in: ", savedir)
    if path.exists(savedir) == False:
        makedirs(savedir, exist_ok=True)
        
    # Initialise the network
    model = DilatedCNN(hidden_size=64, depth=12)
    model = model.cuda()
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Reduce learning rate for last 250 epochs: 
    scheduler = StepLR(optimizer, step_size=(NEPOCHS-250), gamma=0.1)
    
    # Preallocate log arrays
    logging_variables = LogVars(NEPOCHS)        # Log object for tracking losses
    
    for epoch in range(NEPOCHS):
           
        epoch_losses = BubbleLosses()          # Log object for losses
        
        #%% TRAINING
        model.train()
        optimizer.zero_grad()
        for it, sample_batched in enumerate(trn_dataloader):
            
            V = sample_batched['x'].cpu().numpy()    # RF signals
            
            noiselevel = np.float64(random.uniform(low_noise_p, high_noise_p))*V_ref / 100
            
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
            
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            
            # Forward pass
            z = model(V)                        # Predicted bubble distribution
            
            # Compute loss (regression loss, classification loss, and total loss)    
            loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
            
            # Update training losses log object
            epoch_losses.update_trn_metrics(loss_r,loss_b,loss)
        
            # Backpropagation
            loss.backward()
          
        # Update network parameters:
        optimizer.step()
        
        #%% VALIDATION
        model.eval()
        for it, sample_batched in enumerate(val_dataloader):
            
            V = sample_batched['x'].cpu().numpy()    # RF signals
            
            noiselevel = np.float64(random.uniform(low_noise_p, high_noise_p))*V_ref / 100
    
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
            
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            
            # Forward pass
            z = model(V)                        # Predicted bubble distribution
            
            # Compute loss (regression loss, classification loss, and total loss)       
            loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
            
            # Update validation losses log object
            epoch_losses.update_val_metrics(loss_r,loss_b,loss)
        
        scheduler.step()
        
        #%% LOGGING AND SAVING
        # Divide cumulative losses by length of dataloader
        epoch_losses.normalize(len(trn_dataloader), len(val_dataloader))
        
        # Update logging arrays and print log message  
        logging_variables.update(epoch, NEPOCHS, epoch_losses)
        
        # Save model and logging arrays
        modelfile = 'epoch_' + str(epoch)
        modelpath = savedir + delim + modelfile
        torch.save(model.state_dict(), modelpath)
        logging_variables.save(savedir)
    
def train_noise_absolute_range(pulse_name, noiselevel):
    # Train the network on an absolute noise value. This is not a fraction of V_ref, but just the raw values

    print(noiselevel)
    torch.cuda.empty_cache()
    print('Training network on {} with an absolute noiselevel of {}'.format(pulse_name, noiselevel))
    
    #%% SYSTEM SETTINGS
    delim = "\\"
    
    #%% TRAINING PARAMETERS
    NEPOCHS = 1250      # Number of epochs
    NTRN = 1024         # Number of training samples
    NVAL = 960          # Number of validation samples
    BATCH_SIZE = 64     # Batch size
    
    epsilon1 = 1        # Proportionality constant soft loss
    epsilon2 = 1.6      # Proportionality constant Dice loss
    a = 0.1             # Width paramater gaussian convolution kernel  
    
    #%% FILE DIRECTORIES

    trndir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\TRAINING"
    valdir = "D:\\SRML-1D-pulse-types\\Results\RF signals\\txt_files\\" + pulse_name + "\\VALIDATION"
    
    # Exclude non-RF datafiles
    trnfilelist = listdir(trndir)
    idxRF = [index for index, string in enumerate(trnfilelist) if 'RF' in string]
    trnfilelist = [trnfilelist[i] for i in idxRF]
    
    valfilelist = listdir(valdir)
    idxRF = [index for index, string in enumerate(valfilelist) if 'RF' in string]
    valfilelist = [valfilelist[i] for i in idxRF]
    
    
    #%% LOAD THE DATASETS
    trn_ind = np.arange(0,NTRN,1)   # File indices training data   
    val_ind = np.arange(0,NVAL,1)   # File indices validation data
        
    trn_dataset = load_dataset(trndir,trnfilelist,trn_ind)  # Training dataset
    val_dataset = load_dataset(valdir,valfilelist,val_ind)  # Validation dataset
    
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
    
    #%% PATHS
    savedir   = "D:\\SRML-1D-pulse-types\\Results\\Networks"  + delim + "model_" + pulse_name + '_absolutenoise' + str(noiselevel) + delim + str(NEPOCHS) + "_epochs"
    
    print("Weights stored in: ", savedir)
    if path.exists(savedir) == False:
        makedirs(savedir, exist_ok=True)
        
    # Initialise the network
    model = DilatedCNN(hidden_size=64, depth=12)
    model = model.cuda()
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Reduce learning rate for last 250 epochs: 
    scheduler = StepLR(optimizer, step_size=(NEPOCHS-250), gamma=0.1)
    
    # Preallocate log arrays
    logging_variables = LogVars(NEPOCHS)        # Log object for tracking losses
    
    for epoch in range(NEPOCHS):
           
        epoch_losses = BubbleLosses()          # Log object for losses
        
        #%% TRAINING
        model.train()
        optimizer.zero_grad()
        for it, sample_batched in enumerate(trn_dataloader):
            
            V = sample_batched['x'].cpu().numpy()    # RF signals
            
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
            
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            
            # Forward pass
            z = model(V)                        # Predicted bubble distribution
            
            # Compute loss (regression loss, classification loss, and total loss)    
            loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
            
            # Update training losses log object
            epoch_losses.update_trn_metrics(loss_r,loss_b,loss)
        
            # Backpropagation
            loss.backward()
          
        # Update network parameters:
        optimizer.step()
        
        #%% VALIDATION
        model.eval()
        for it, sample_batched in enumerate(val_dataloader):
            
            V = sample_batched['x'].cpu().numpy()    # RF signals
    
            # Add noise to the RF signals and convert to torch cuda tensor:
            V = add_noise(V,noiselevel,filt_b,filt_a)
            
            y = sample_batched['y1'].cuda()     # Ground truth bubble distribution
            
            # Forward pass
            z = model(V)                        # Predicted bubble distribution
            
            # Compute loss (regression loss, classification loss, and total loss)       
            loss_r, loss_b, loss = dual_loss(z,y,epsilon1,epsilon2,a)
            
            # Update validation losses log object
            epoch_losses.update_val_metrics(loss_r,loss_b,loss)
        
        scheduler.step()
        
        #%% LOGGING AND SAVING
        # Divide cumulative losses by length of dataloader
        epoch_losses.normalize(len(trn_dataloader), len(val_dataloader))
        
        # Update logging arrays and print log message  
        logging_variables.update(epoch, NEPOCHS, epoch_losses)
        
        # Save model and logging arrays
        modelfile = 'epoch_' + str(epoch)
        modelpath = savedir + delim + modelfile
        torch.save(model.state_dict(), modelpath)
        logging_variables.save(savedir)
        
if __name__ == "__main__":
    
    # Parse the pulse name
    parser = argparse.ArgumentParser()
    parser.add_argument("pulse_name", help="Specify the pulse name of the dataset" )
    parser.add_argument('--noise', action=argparse.BooleanOptionalAction)
    parser.add_argument('--abs_noise', help="Absolute noise value for training", type = float, nargs='?', default=None)
    parser.add_argument('--low_noise_p', help="Lower percentage noise value for training", type = float, nargs='?', default=None)
    parser.add_argument('--high_noise_p', help="Upper percentage value for training", type = float, nargs='?', default=None)
    args = parser.parse_args()
    
    pulse_name      = args.pulse_name
    noise           = args.noise
    low_noise_p     = args.low_noise_p
    high_noise_p    = args.high_noise_p
    abs_noise       = args.abs_noise
    print(pulse_name,noise,low_noise_p,high_noise_p,abs_noise)
    
    if low_noise_p is not None and high_noise_p is not None:
        train_noise_range(pulse_name, low_noise_p, high_noise_p)
    elif abs_noise is not None:
        train_noise_absolute_range(pulse_name, abs_noise)
    elif noise == True:
        train_noise(pulse_name)
    else:
        train(pulse_name)