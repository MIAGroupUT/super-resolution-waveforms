# Load the data

import os
import numpy as np
import torch

def load_data(filedir,filelist,ind):
    """
    Load the bubble count data, the RF data, and into separate 2D arrays.
    """

    count = 0
    
    for n in ind:
        
        filename = filelist[n]
        bubbles  = []           # Bubble count array
        RFline   = []           # RF array
    
        # Read the text file line by line:
        with open(os.path.join(filedir,filename)) as f:
            
            for k,line in enumerate(f.readlines()):
                if k>3:
                    values = line.split(',')            
                    bubbles.append(float(values[0]))
                    RFline.append(float(values[1]))   
    
        # Add the 1D arrays to the 2D array. For the first row, create the 2D
        # array.
        if count == 0:
            bubbleData   = np.array([bubbles])
            RFlinesData = np.array([RFline])
            count = count + 1
        else:
            bubbleData   = np.concatenate((bubbleData,[bubbles]))
            RFlinesData = np.concatenate((RFlinesData,[RFline]))
            
    return RFlinesData, bubbleData

def load_descriptive_dataset(filedir):
    filedir

class BubbleDataset(torch.utils.data.Dataset):

    def __init__(self, RFlines, locations):
        self.RFlines   = RFlines
        self.locations = locations
    
    def __len__(self):
        return self.RFlines.shape[0]
    
    def __getitem__(self, idx):
        sample = {'x':  self.RFlines[idx, :].unsqueeze(0),
                  'y1': self.locations[idx, :]} 
        return sample

class DescriptivesDataset:
    def __init__(self, Nb, PA, computer, t_tot, t_pulse):
        self.Nb         = Nb
        self.PA         = PA
        self.computer   = computer
        self.t_tot      = t_tot
        self.t_pulse    = t_pulse
    
    def __len__(self):
        return self.Nb.shape[0]
    
    def __getitem__(self, idx):
        sample = {
            'Nb': self.Nb[idx],
            'PA': self.PA[idx],
            'computer': self.computer[idx],
            't_tot': self.t_tot[idx],
            't_pulse': self.t_pulse[idx]
        }
        return sample


def load_dataset(filedir,filelist,ind):
    
    RFlinesData, bubblesData = load_data(filedir,filelist,ind)
       
    dataset = BubbleDataset(
        torch.from_numpy(RFlinesData).float(),
        torch.from_numpy(bubblesData).float())
        
    return dataset

def load_descriptives(filedir):
    Nb = []
    PA = []
    computer = []
    t_tot = []
    t_pulse = []
    
    filename = "simulationDescriptives.txt"
    
    # Read the text file line by line:
    with open(os.path.join(filedir,filename)) as f:
        
        for k,line in enumerate(f.readlines()):
            if k>1:
                values = line.split(',')            
                Nb.append(int(values[0]))
                PA.append(float(values[1]))
                computer.append(str(values[2]))
                t_tot.append(float(values[3]))
                t_pulse.append(float(values[4]))
    

    Nbdata = np.array([Nb])
    PAdata = np.array([PA])
    computerdata = np.array([computer])
    t_totdata = np.array([Nb])
    t_pulsedata = np.array([t_pulse])
    
    descriptiveData = DescriptivesDataset(
        Nbdata,
        PAdata,
        computerdata,
        t_totdata,
        t_pulsedata)
    
    return descriptiveData
            
                
    
    
