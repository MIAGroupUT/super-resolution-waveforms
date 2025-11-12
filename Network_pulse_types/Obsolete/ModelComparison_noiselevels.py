""" 
Compare different models over different noiselevels
"""

# Import the models
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, makedirs
import torch
from scipy import signal

from bubblenetwork import DilatedCNN
from bubblenetwork import RandomModel
from bubbledataloader import load_dataset
from bubblemetrics import spatial_tolerant_stats
from customModelInfo import model_info

#%% INPUTS
delim = "\\"

figWidth    = 350/25.4  # Figure width
figHeight   = 150/25.4  # Figure height
lineWidth   = 2.5
fontSize    = 20
labelSize   = 16

NEPOCHS = 1250 

# Evaluated models
model_list = ["pulseSingle_Reference_OneCycle","pulseChirp_Long_Downsweep","pulseChirp_Short_Downsweep","pulseSingle_Short_MedF","pulseSingle_Long_MedF"]

#%% DATA SET PARAMETERS
NDATA = 960         # Number of data files
BATCH_SIZE = 8      # Batch size

#%% NOISE

V_ref = 9.65                                # Reference value noise
noiselevels_p = np.array([4,8,16,32,64,128,256],dtype='int32')       # Noise level percentage
noiselevels   = noiselevels_p*V_ref/100       # Noise level

# Filter parameters
fs = 62.5   # Sampling frequency (MHz)
n = 4       # Order of the butterworth filter
fc = 1.7*3  # Cut-off frequency (MHz)

# low-pass filter coefficients
filt_b, filt_a = signal.butter(n, 2*fc/fs, 'low')

def add_noise(V,sigma,b,a):
    # Add noise to the signals and convert to torch cuda tensor
    
    mu = 0  # Mean value of the random distribution
    
    # Add noise to the signal and apply low-pass filter
    V_noise = V + np.random.normal(mu,sigma,size = V.shape)
    V_noise_filt  = signal.filtfilt(b,a,V_noise,axis=-1)
    
    # Convert the RF signals to torch cuda tensor
    V_noise_filt = torch.from_numpy(V_noise_filt.copy())
    V_noise_filt = V_noise_filt.type(torch.FloatTensor)
    V_noise_filt = V_noise_filt.cuda()
    
    return V_noise_filt

#%% LOOP THROUGH THE MODELS
for k,modelname in enumerate(model_list):
    print(modelname)  
    
    model_properties = model_info[modelname]
    
    for j,noiselevel in enumerate(noiselevels):
        
        
        #%% FILE DIRECTORIES
        datadir     = "D:\\SRML-1D-pulse-types\\Results\\RF signals\\txt_files\\" + modelname + "\\TESTING"
        filelist    = listdir(datadir)   
        
        modeldir    = "D:\\SRML-1D-pulse-types\\Results\\Networks\\model_" + modelname + "_noise" + str(noiselevels_p[j]) + delim + str(NEPOCHS) + "_epochs"
        
        savedir = "D:\\SRML-1D-pulse-types\\Results\\Figures"
        
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
        epoch = NEPOCHS-1
        
        if modelname == 'R':
            model = RandomModel()
        
        else:   
            model = DilatedCNN(hidden_size=64, depth=12)  
            modelpath = modeldir + delim + 'epoch_' + str(epoch)
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
                V = sample_batched['x'].cpu().numpy()    # RF signals
        
                # Add noise to the RF signals and convert to torch cuda tensor:
                V = add_noise(V,noiselevel,filt_b,filt_a)
                
                y = sample_batched['y1'].cuda()     # Ground truth
                
                # Forward pass
                z = model(V)                        # Prediction
                
                # Compute statistics on the batch:
                TP,FN,FP,P,R = spatial_tolerant_stats(y,z,tolerance,threshold) 
                
                P[P==0] = 1e-3     # Prevent divide by 0
                R[R==0] = 1e-3     # Prevent divide by 0
    
                # Collect F1 score each prediction in a list:
                F1        = np.append(F1,2*P*R/(P+R))
    
            F1_array[i] = np.mean(F1)
        np.save(modeldir + delim + "F1_test_" + modelname,F1_array) 
    
    # Initialize the figure
    
        if j == 0:
            fig = plt.figure(figsize=(figWidth,figHeight))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            
        ax1.plot(F1_array,color = model_properties['color'], linestyle = model_properties['linestyle'], linewidth = lineWidth, label = model_properties['abbreviation'] +"n"+ str(noiselevels_p[j]))


    ax1.set_xlabel('tolerance (grid points)',fontsize=fontSize,  family='arial')
    ax1.set_ylabel('F1 score',fontsize=fontSize,  family='arial')
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,9])
    ax1.tick_params(labelsize=labelSize)
    ax1.grid()
    
    # Make the legend
    lgd = ax1.legend(fontsize=fontSize, loc='right')
    lgd.get_texts()[0].set_fontfamily('Arial')
    
    # Set the ticks and labels for the upper x-axis (ax2)
    ax2.grid()
    ax2.set_xlabel('tolerance (microm)', fontsize = fontSize, family='arial' )
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(np.int16(ax1.get_xticks()*(1500*16e-9)/1e-6))
    ax2.tick_params(labelsize=labelSize)

    plt.savefig(savedir + delim + "ModelComparison_noise_" + modelname + ".svg")
