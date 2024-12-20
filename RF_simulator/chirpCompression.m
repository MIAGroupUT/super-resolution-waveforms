% Chirp compression
%
% This script opens the chirp RF simulations, compresses the voltage lines
% and saves the compressed files in a seperate folder.
%
% Rienk Zorgdrager, University of Twente, 2024

%% Apply the matched filter on the chirp signals

clear; clc; close all

delim = "\";
parentdir = "D:\SRML-1D-pulse-types\RF signals\RESULTS";

rescaling = true;

%% Simulation properties
dispFig = false;            % Display figures

% Add the functions folder to path
addpath './functions'
addpath './microbubble-simulator/functions'

% Get the transducer transfer functions of the P4-1 transducer
Tfit = load('TransmitTransferFunctionFit.mat');
Hfit = load('ReceiveTransferFunctionFit.mat','Hfit');

pulseProperties = get_pulse_properties();

% Simulation and recording sampling rate:
Fs = Tfit.Fs;              	% Simulation sampling rate (Hz)
downsample = 4;             % Measurement sampling rate: Fs/downsample
pulseProperties.downsample = downsample;
pulseProperties.Fs = Fs;

[pulses, pulseSequence] = getPulseInOutput(pulseProperties, Tfit, dispFig);

%% Index the chirps and get the chirp names
pulseNames = cell(1,length(pulseSequence));

for i = 1:length(pulseSequence)
    pulseCat = pulseSequence(i).pulseCategory;
    lenCat   = pulseSequence(i).lengthCategory;
    varCat   = pulseSequence(i).variableCategory;

    pulseNames{i} = string(pulseCat+"_"+lenCat+"_"+varCat);
end

chirpNames = pulseNames(contains(string(pulseNames),"Chirp"));

%% Compress the chirps
for j = 1:length(chirpNames)
    
    chirpName = join(split(chirpNames{j},'_'));

    categories = split(string(chirpNames(j)),"_");
    pulseCat = categories{1};
    lenCat   = categories{2};
    varCat   = categories{3};

    disp(chirpName)
    filedir = parentdir + delim + string(chirpNames(j));
    save_dir = filedir + "_compressed";

    % Make new directory
    if ~exist(save_dir,'dir')
        mkdir(save_dir)
    end
    
    p_norm = pulses.(pulseCat).(lenCat).(varCat).p_norm;
    
    % Create matched filter
    
    % Convolve pressure wave with receiver transfer function:
    V_pulse = receiveTransferFunction(p_norm, Hfit.Hfit,Fs);
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
    
    % Load the new files
    fileNames = dir(filedir);
    fileNames = fileNames(3:end);

    for sim = 1:length(fileNames)
        filename = fileNames(sim).name;
        load(filedir+delim+fileNames(sim).name)

        % Apply the filter in the RF line
        RF.V = conv(RF.V,matched_filter)/A;
        RF.V(1:(N-1)) = []; % Cut off the zero-padded start of convolution

        % Save the RF line
        save(strcat(save_dir,delim,filename),...
            'domain', 'liquid','gas','shell','pulse',...
            'bubble','RF')
     end
end