%% Determines the noise level and SNR in the measurements
% This computes the SNR level. Furthermore, it generates figures which can
% be used to study the acquisition noise captured in receive-only 
% measurements.
%
% Rienk Zorgdrager, University of Twente, 2024

%% INPUTS
% Delimiter 
delim = "\\";

% Pulse name
pulse_name = "pulseExpVal_Short_chirp"; % Enter "pulseExpVal_Short_SIP" or "pulseExpVal_Short_chirp"

% Path to signal folder. This is an acquisition with an extreme number of
% bubbles in the signal.
parent_path = "D:\SRML-1D-pulse-types\Results\Experiments\Data\Processed\Noise estimation\Signal\20240805_exp_val_data_2-maxbubbles_TGCmax_voltage-2.5\";

% Path to noise folder.
path = "D:\SRML-1D-pulse-types\Results\Experiments\Data\Processed\Noise estimation\Noise";

% Measurement depth of evaluation
z = 0.05;       % average evaluation depth in m
th = 0.05;      % half width of the range in m

%% ADD EXPORT_FIG
addpath('C:\Users\rienk\OneDrive\Bureaublad\export_fig-master')

%% OTHER SETTINGS

% Parameters
c = 1480;       % speed of sound in m/s
Fs = 62.5e6;    % sampling frequency in Hz

% Obtain start and end samples of the evaluation segment
s1 = int32(2*((z - th) / c) * Fs)+1;    % First boundary
s2 = int32(2*((z + th) / c) * Fs);      % Second boundary

%% PREPARE NOISE INVESTIGATION
% Store the files in a directory
directory = dir(path);
directory = directory(3:end);

% Initialize data structures and arrays
measurement_avg = struct();
noise_data = zeros(10,96*8446);

% Get the frame numbers
if pulse_name == "pulseExpVal_Short_SIP"
    frames = 1:2:19;
elseif pulse_name == "pulseExpVal_Short_chirp"
    frames = 2:2:20;
else
    msg = "Incorrect pulse name. Enter 'pulseExpVal_Short_SIP' or 'pulseExpVal_Short_chirp'";
    error(msg)
end

%% LOOP THROUGH THE FILES
for file = 1:length(frames)
    
    % Load signal part
    load(parent_path + pulse_name + delim +sprintf('RFDATA%05d',frames(file)))
    RF_signal = RF;
    
    % Load noise
    load(directory.folder + delim + directory.name + delim + pulse_name + delim + sprintf('RFDATA%05d',frames(file)))

    for el = 1:length(RF)
        %% DETERMINE THE NOISE
        % Compute the root mean square of the noise
        segment = RF(el).V(s1:s2);
        V_noise(el) = rms(segment);

        fprintf('Noise level is %f V \n', V_noise(el));

        %% DETERMINE THE SIGNAL
    
        % Compute the root mean square of the signal
        segment = RF_signal(el).V(s1:s2);
        V_signal(el) = rms(segment);
    
        fprintf('Signal level is %f V \n', V_signal(el));

        %% DETERMINE THE SNR
        V_SNR(el) = (V_noise(el)/V_signal(el))*100;

    end
    %% STORE OUTCOMES
    % Store the values in the struct
    measurement_avg(file).Noise = mean(V_noise);
    measurement_avg(file).Signal = mean(V_signal);
    measurement_avg(file).SNR = mean(V_SNR);
    
    % Store individual values for the histogram
    noise_data(file,:)= [RF(:).V]; 
end

% Reshape the data
noise_data = reshape(noise_data,[1,size(noise_data,1)*size(noise_data,2)]);

%% DISPLAY AND PLOT THE RESULTS
% Display V_ref
V_SNR = (V_noise/V_signal)*100;

fprintf('Noise fraction is %f % \n', V_SNR);

% Plot the noise level per measurement
f = figure('Units', 'inches', 'Position', [1 1 3.5 2]);
bar([measurement_avg.Noise], 0.4)
ylabel("RMS Noise [ADC units]")
xlabel("Measurement")
title("Noise level per measurement")
f.Color = 'w';

% Export the figure
export_fig MeasurementNoise.pdf

% Plot the distribution 
fhist = figure();
histfit(noise_data);
xlabel('Noise value [ADC]');
ylabel('Counts');
legend('datapoints', 'fit');

% Export the figure
export_fig noiseDistribution.pdf