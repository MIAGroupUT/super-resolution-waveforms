% Delay-and-sum reconstruction of the bubble image from the element data
% simulated with RF_simulatorFINAL. After super-resolving the RF data.
%
% Nathan Blanken, University of Twente, 2021
% Adjusted by Rienk Zorgdrager, University of Twente, 2024

clear
clc

delim = '\';

additional_specification = "_noise128";

%% FIGURE SETTINGS
dpi = 600;

%% LOAD DATA AND METADATA
folderpath = 'D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D';
directory = dir(folderpath);
directory = directory(3:end);

%% FILTER THE DIRECTORIES IF NEEDED

if contains(additional_specification,"_noise") % Select the models to be evaluated for noise
    filt1 = contains({directory.name},'pulseSingle_Reference_OneCycle_500bubbles');
    filt2 = contains({directory.name},'pulseSingle_Short_MedF_500bubbles');
    filt3 = contains({directory.name},'pulseSingle_Long_MedF_500bubbles');
    filt4 = contains({directory.name},'pulseChirp_Long_Downsweep_500bubbles');
    filt5 = contains({directory.name},'pulseChirp_Short_Downsweep_500bubbles');
    combinedFilter = boolean(filt1+filt2+filt3+filt4+filt5);
    directory = directory(combinedFilter);
end

for j=1:length(directory)
    % Define directories
    dirpath = string(directory(j).folder) + string(delim) + string(directory(j).name);
    filepath =  dirpath + string(delim) + "RFDATA00001.mat";

    % Load the metadata from the original MATLAB files:
    load(filepath)

    % Domain and transducer properties
    width = domain.width;       % domain width (m)
    depth = domain.depth;       % domain depth (m)
    N = length(RF(1).p);        % number of samples per RF line
    Nelem = length(RF);         % number of transducer elements
    Fs = RF(1).fs;              % sample rate (Hz)
    t = (0:(N-1))/Fs;           % time axis (s)

    x_el = linspace(-width/2,width/2,Nelem); % Element positions (m)
    
    % Load the super-resolved RF data:
    filename_sr = strrep(filepath, '.mat', "_sr"+ additional_specification+".txt");
    RF_matrix   = read_rf_txt(filename_sr,Nelem);
    RF_matrix   = RF_matrix';

    %% WAVE PROPERTIES
    c = liquid.c;               % speed of sound in the medium (m/s)
    f0 = pulse.f0;              % centre frequency (Hz)
    lambda = c/f0;              % wavelength (m)

    %% TIME GAIN COMPENSATION (TGC)
    RF_TGC = RF_matrix;

    %% DELAY-SUM RECONSTRUCTION
    sig_dur = 0;             	% Signal duration (samples)

    % Dimensions of the reconstructed image:
    IM_WI = width*1.5;          % Width of the reconstructed image (m)
    IM_DE = depth;              % Depth of the reconstructed image (m)
    pix_siz = lambda/50;        % Pixel size for image reconstruction

    % Delay and sum reconstruction:
    tic
    img = delay_and_sum(RF_TGC, IM_WI, IM_DE, pix_siz, x_el, c, sig_dur, Fs);
    toc

    %% SAVE THE RESULTS
    x = -IM_WI/2:pix_siz:IM_WI/2;       % Lateral coordinates (m)
    z = 0:pix_siz:IM_DE;                % Axial coordinates (m)
    
    save(dirpath+delim+directory(j).name+"DAS_sr"+ additional_specification+".mat",'img','x','z')
    img_path = dirpath+delim+directory(j).name+"DAS_sr"+ additional_specification+".pdf";
    exportgraphics(gcf, img_path, 'ContentType', 'vector', 'Resolution', dpi)
end

