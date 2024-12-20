# Visualisation and inspection of the statistics obtained on the test
# set of DATA_FINAL with StatisticsNumberBubbles.py.
# StatisticsNumberBubbles.py and were executed on the GPU server. The 
# statistics were stored in .npy files, which were copied back to the C drive 
# for futher inspection.
#
# F1 score is given as a function of the number of bubbles, and 
# the transmitted acoustic pressure. Scatter plots with one point per
# prediction.

import numpy as np
import matplotlib.pyplot as plt
from customModelInfo import model_info
from os import listdir

#%% INPUTS
delim = "\\"

# Figure parameters
figWidth    = 3.5
figHeight   = 2.5#figWidth*(3/4)
lineWidth   = 1
fontSize    = 8
labelSize   = 8
dpi = 600
fontFamily  = "Times New Roman" 

NEPOCHS = 1250
tols = [1,4]

# Store the results in this directory
savedir     = "D:\\SRML-1D-pulse-types\\Results\\Figures"

# Set rcParams
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = fontFamily

#%% GET THE MODELS
model_list = list(model_info.keys())

# Get the networks to be evaluated
network_dir = "D:\\SRML-1D-pulse-types\\Results\\Networks"
networks = listdir(network_dir)

# Loop through the tolerances
for j,tol in enumerate(tols):
    
    for k,modelname in enumerate(networks):
        
        print(modelname)
        if "noise" in modelname:
            noise = ''.join(filter(str.isdigit, modelname))
            noiselevel_p = int(noise)
            
            # Get the right model name for the raw data
            parent_model = modelname.replace('model_','')
            parent_model = parent_model.replace(noise,'')
            parent_model = parent_model.replace('_noise','')
            print("Noise level: " + str(noiselevel_p))
            print('This script is not intended for noise networks, please use another script to investigate noise')
            continue
        elif "linear" in modelname:
            print("Simulation descriptives not present as they were not stored. Please rerun the simulations and store the simulation settings.")
            continue
        else:
            parent_model = modelname.replace('model_','')
        model_properties = model_info[parent_model]
        
        
        
        #%% FILE DIRECTORIES
        
        # Investigate this model:
        modeldir    = network_dir + delim + modelname  + delim + str(NEPOCHS) + '_epochs'
        statisticsdir = modeldir + delim + 'StatisticsEvaluation'
        
        # Get optimal threshold values and the corresponding tolerance list:
        th_opt    = np.load(modeldir + delim + 'thresholds_optimal.npy')
        tol_list  = np.load(modeldir + delim + 'tolerance_list.npy')
        
        F1_matrix = np.load(statisticsdir + delim + 'F1_matrix.npy')   
        NB_matrix = np.load(statisticsdir + delim + 'num_bub_matrix.npy') 
        PA_matrix = np.load(statisticsdir + delim + 'pressure_matrix.npy') 
    
    
        #%% CREATE PLOTS
        tol_idx = np.where(tol_list == tol)  # corresponding index in tolerance list
      
        score   = np.squeeze(F1_matrix[j,:])
        ylabel_str = 'F1 score'
            
        # Optimal threshold for model and tolerance:
        threshold =   th_opt[tol_idx]
        
        #%% Number of bubbles
        Nmax = 1000
        Nmin = 10
        
        # Define bins
        binSize     = round((Nmax-Nmin)/np.sqrt(len(score))) # Maybe go to 40
        bins        = np.arange(Nmin,Nmax+binSize,binSize)
        bins_ctr    = bins - 0.5*binSize
        
        bin_indices = np.digitize(NB_matrix,bins)
        
        # Create a dictionary to store each data by bin
        # Compute the means and the standard deviation
        means   = [np.mean(score[bin_indices == bin_num]) for bin_num in range(1,len(bins))]
        std_dev = [np.std(score[bin_indices == bin_num]) for bin_num in range(1,len(bins))]
        
        # Plot the figures
        NB_str = 'Number of bubbles (tolerance = %d)' % (tol)
        
        # Initialize the figure
        if k == 0:
            fig_NB, ax_NB = plt.subplots(1,1, figsize=(figWidth,figHeight), dpi = dpi)
            ax_NB.grid()
            ax_NB.set_ylim([0.35,1])
            ax_NB.set_xlim([0,1000])
            ax_NB.set_title(NB_str,fontsize=fontSize,family=fontFamily)
            ax_NB.set_ylabel('F1 score (mean ± std)',fontsize=fontSize,family=fontFamily)
            ax_NB.set_xlabel('Number of bubbles',fontsize=fontSize,family=fontFamily)
            ax_NB.tick_params(labelsize=labelSize)
        
        ax_NB.errorbar(bins_ctr[1:], means, color = model_properties['color'], linestyle=model_properties['linestyle'], linewidth = lineWidth, label=model_properties['abbreviation'])
        #%% Acoustic pressures
        Pmin = 5e3 #Pa
        Pmax = 250e3 #Pa
        
        # Define bins
        binSize     = round((Pmax-Pmin)/np.sqrt(len(score))) # Maybe go to 40
        bins        = np.arange(Pmin,Pmax+binSize,binSize)
        bins_ctr    = bins - 0.5*binSize
        
        bin_indices = np.digitize(PA_matrix,bins)
        
        # Create a dictionary to store each data by bin
        # Compute the means and the standard deviation
        means   = [np.mean(score[bin_indices == bin_num]) for bin_num in range(1,len(bins))]
        std_dev = [np.std(score[bin_indices == bin_num]) for bin_num in range(1,len(bins))]
        
        # Plot the figures
        title_str = 'Transmitted acoustic pressure (tolerance = %d)' % (tol)
        
        
        if k == 0: # Initialize the figure
            fig_PA, ax_PA = plt.subplots(1,1, figsize=(figWidth,figHeight), dpi = dpi)
            ax_PA.grid()
            ax_PA.set_ylim([0.35,1])
            ax_PA.set_xlim([0,Pmax/(1e3)])
            ax_PA.set_ylabel('F1 score (mean ± std)',fontsize=fontSize,family=fontFamily)
            ax_PA.set_xlabel('Transmitted acoustic pressure (kPa)',fontsize=fontSize,family=fontFamily)
            ax_PA.tick_params(labelsize=labelSize)
    
        ax_PA.plot(bins_ctr[1:]/(1e3), means, color = model_properties['color'], linestyle=model_properties['linestyle'], linewidth = lineWidth, label=model_properties['abbreviation'])
    
    # Insert legend and export
    figname_NB = 'Figure_F1_vs_NB_tol%d.svg' % tol
    ax_NB.legend(loc='right', fontsize=fontSize)
    fig_NB.savefig(savedir + delim + figname_NB)
    
    figname_PA = 'Figure_F1_vs_PA_tol%d.svg' % tol
    ax_PA.legend(loc='right', fontsize=fontSize)
    fig_PA.savefig(savedir + delim + figname_PA)
