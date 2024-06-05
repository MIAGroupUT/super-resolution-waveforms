% Delay-and-sum reconstruction of the bubble image from the element data
% simulated with RF_simulatorFINAL.
%
% Nathan Blanken, University of Twente, 2021
% Adjusted by Rienk Zorgdrager, University of Twente, 2024


clear
clc

delim = '\';

%% ADD PATHS
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\RF_simulator\functions');
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\RF_simulator\extra_files');

%% LOAD DATA AND METADATA
Hfit = load('ReceiveTransferFunctionFit.mat','Hfit');

folderpath = 'D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D';
directory = dir(folderpath);
directory = directory(3:end);

% For trial:
directory = directory(1:16);

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
    
    % Compress the chirps
    if contains(directory(j).name,"Chirp") == true
        disp('Chirp is being compressed')
        RF = compressChirp(pulse, RF, Hfit);
    end

    % Convert RF struct to matrix
    RF_matrix = [RF.V];
    RF_matrix = reshape(RF_matrix, N, Nelem)';  % matrix of element RF data

    %% WAVE PROPERTIES
    c = liquid.c;               % speed of sound in the medium (m/s)
    f0 = pulse.f;               % centre frequency (Hz)
    lambda = c/f0;              % wavelength (m)

    %% TIME GAIN COMPENSATION (TGC)
    % Apply a linear TGC to compensate for the 1/r decay of scattered pressure:
    TGC = t;

    RF_TGC = RF_matrix;
    for i = 1:Nelem
        RF_TGC(i,:) = RF_matrix(i,:).* TGC ;
    end
    clear RF_matrix

    %% DELAY-SUM RECONSTRUCTION
    % Compute the approximate duration of the pulse in number of samples:

    two_way_pulse = receiveTransferFunction(pulse.p_norm, Hfit.Hfit, pulse.fs);
    [~,I] = max(abs(hilbert(two_way_pulse)));       % Find maximum of pressure pulse
    %sig_dur = I/pulse.fs*RF(1).fs;                  % Signal duration (samples)
    
    if contains(directory(j).name,"Chirp") == true
        sig_dur = 0;
    else
        sig_dur = I/pulse.fs*RF(1).fs;
    end

    % [~,I] = max(abs(hilbert(pulse.p_norm)));     % Find maximum of pressure pulse
    % sig_dur = I/pulse.fs*RF(1).fs*2;        % Signal duration (samples)
    clear RF

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
    
    save(dirpath+delim+directory(j).name+'DAS.mat','img','x','z')
end

function RF = compressChirp(pulse, RF, Hfit)

    rescaling = true; % Rescale to original pulse amplitude

    p_norm = pulse.p_norm;
    downsample = pulse.fs / RF(1).fs;

    % Compress chirps
    % Convolve pressure wave with receiver transfer function:
    V_pulse = receiveTransferFunction(p_norm, Hfit.Hfit, pulse.fs);
    V_pulse = V_pulse/max(abs(V_pulse)); % Normalise expected pulse
    V_pulse = V_pulse(1:downsample:end); % Downsample the expected pulse
    matched_filter = conj(fliplr(V_pulse));
    N = length(matched_filter);

    % Rescaling factor to scale the amplitude of the compressed signal back
    % to the order of magnitude of the amplitude of the uncompressed
    % signal:
    if rescaling == true
        A = max(conv(V_pulse,matched_filter));
    else
        A = 1;
    end

    for el = 1:length(RF)
        % Apply the filter in the RF line
        RF(el).V = conv(RF(el).V,matched_filter)/A;
        RF(el).V(1:(N-1)) = []; % Cut off the zero-padded start of convolution
    end
end