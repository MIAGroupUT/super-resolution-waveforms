%% Process experimental data 
% This script transforms the Verasonics Outputs to .mat files which can be
% used as input for the neural network.
%
% It scales the RF signals by differences in the transfer function
% sensitivity between measurements and simulations and the TGC used in
% measurements. To do so, it requires a maximized TGC.
%
% Rienk Zorgdrager, University of Twente, 2024

%% CLEAR
clear

%% INPUTS
delim = '\\';
addpath("extra_files\")

% Directory to raw data files
expdir = 'D:\SRML-1D-pulse-types\Results\Experiments\Data\Raw';
[filename,filedir] = uigetfile(expdir, 'Select the datafile');

%% LOAD DATA FILE AND OTHER CONNECTOR SPECIFICATIONS
% Load data file
load(strcat(filedir,filename))

% Dataset name
datasetname = erase(filename,".mat");

% Determine whether it is a noise measurement
if contains(filedir, "\Noise")
    noise = true;
else
    noise = false;
end

% Define the pulse names
pulsenames = ["pulseExpVal_Short_SIP", "pulseExpVal_Short_chirp"];

% Obtain the channels for all elements
connectorArray = Trans.Connector;

%% COMPENSATE FOR TRANSFER FUNCTION DIFFERENCES AND GAIN FOR SCALING
% Simulations TF
Rcv_TF_sim = load("C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\RF_simulator\microbubble-simulator\functions\ReceiveTransferFunctionFit.mat");
Hfit_sim = Rcv_TF_sim.Hfit;
Fs = Rcv_TF_sim.Fs;
N_sim = length(Hfit_sim);
f_sim = (0:(N_sim-1))*Fs/N_sim;

% Experiments TF
Rcv_TF_exp = load("measured_receive_impulse_response.mat");
IR = Rcv_TF_exp.IR;
N_exp = length(IR);
Hfit_exp = fft(IR,N_sim)/Fs;

% Plot the transfer functions
figure;
yyaxis left
plot(f_sim*1e-6,abs(Hfit_sim));
xlim([0,5])
xlabel('Frequency (MHz)')
ylabel('Sensitivity (Sim. units/Pa)')
yyaxis right
hold on
plot(f_sim*1e-6,abs(Hfit_exp));
xlim([0,5])
xlabel('Frequency (MHz)')
ylabel('Sensitivity (ADCL/Pa)')

% Compute scaling transfer function
TF_scaling = Hfit_sim./Hfit_exp;
TF_gain = mean(abs(TF_scaling(f_sim<5e6))); % in [Sim. units/Verasonics units]. This is a correction for sensitivity differences between transfer functions.

% Check if TGC is maximized
if length(unique(TGC.Waveform)) > 1 % Check if the TGC is the same throughout one RF line
    msg = "This script is not able to process data with varying TGC.  For SR imaging, please use the same TGC throughout the entire RF signal.";
    error(msg)
elseif unique(TGC.Waveform) ~= 1023
    msg = "TGC should be maximized during measurements. Please change the TGC or adjust the script to correct for different gain values.";
    error(msg)
end

%% OTHER GAINS AND FILTERING
% Define a digital filter to remove low F components outside the
% bandwidth of the P4-1 Transducer
fc          = 0.5e6;    % Cut-off frequency in Hz
fv          = linspace(0,Receive.decimSampleRate*1e6,N_sim);
Hfilt       = ones(size(fv));
Hfilt(fv<fc)= 0;
Hfilt(fv>(max(fv)-fc)) =0;

% Define time-dependent ROI gain (this is not TGC, but an arbitrary way to remove artefacts from the signal)
% Define region of interest
ROI1        = 1000;     % in samples
ROI2        = 6000;     % in samples
I           = 200;      % Adjust the signal intensity before and after ROI during I samples

% TGC gain
alpha       = 40; % dB, this is the max TGC gain in the Verasonics Vantage
TGC_gain    = 1/(10^(alpha/20));

%% SEPERATE THE DATA
% Extract the data from the buffers in RcvData

nBuffers = length(RcvData)-1; % Determine the number of buffers
acqBuffers = [2];

for buf = 1:nBuffers
    % Loop through the buffers
    % Get the right buffer

    acqBuffer = acqBuffers(buf);

    % Retrieve the data of the buffer
    data = RcvData{acqBuffer};

    % Extract frames out of the superframes
    n_sFrames = P.numAcqsSuperFrame;                        % Number of frames in a superframe
    sFrameLen = Resource.RcvBuffer(acqBuffer).rowsPerFrame; % Length of a superframe
    frameLen  = Receive.endSample/n_sFrames;                % Length of each frame
    
    % Extract the data out of the super-frames
    data = ExtractFrames(data,n_sFrames,frameLen);

    % Determine the total number of acquisition frames
    nFrames = size(data,3); 

    for frame = 1:nFrames
        % Loop through the frames, they should be seperated per pulse. The
        % uneven pulses are the SIP, the evens are the chirp

        pulse = struct();

        if mod(frame,2) ~= 0 % SIP
            % store the data in the SIP folder
            % Define destination directory
            if noise == true
                dstdir = fullfile(expdir,'Data','Processed',"Noise estimation",datasetname,pulsenames(1));
            else
                dstdir = fullfile(expdir,'Data','Processed',datasetname,pulsenames(1));
            end

            % Make the datastructures
            RF = seperateElements(data,frame,connectorArray);

            % Scale and filter
            RF = scaleAndFilterRF(RF, TF_gain, TGC_gain, N_sim, Hfilt, ROI1, ROI2, I, noise);

            % Construct pulse dataframe
            pulse.p_norm = TW(1).Wvfm1Wy';
            pulse.f = TW(1).Parameters(1)*1e6;
            pulse.Wvfm2Wy = TW(1).Wvfm2Wy';
            pulse.fs = Resource.VDAS.sysClk*1e6;

        else % Chirp
            % store the data in the chirp folder
            % Define destination directory
            if noise == true
                dstdir = fullfile(expdir,'Data','Processed',"Noise estimation",datasetname,pulsenames(2));
            else
                dstdir = fullfile(expdir,'Data','Processed',datasetname,pulsenames(2));
            end

            % Make the datastructures
            RF = seperateElements(data,frame,connectorArray);

            % Scale the RF line
            RF = scaleAndFilterRF(RF, TF_gain, TGC_gain, N_sim, Hfilt, ROI1, ROI2, I, noise);

            % Construct pulse dataframe
            pulse.p_norm = TW(2).Wvfm1Wy';
            pulse.f = median(TW(2).envFrequency)*1e6;
            pulse.Wvfm2Wy = TW(2).Wvfm2Wy';
            pulse.fs = Resource.VDAS.sysClk*1e6;
        end

        % Directories
        filename = sprintf('RFDATA%05d',frame);

        % Create the directory if it does not exist
        if ~exist(dstdir) == 1
            mkdir(dstdir) 
        end

        % Specify file directory
        RFdir = fullfile(dstdir,filename);

        % Save the RF data as a .mat file
        save(RFdir, "RF","pulse")

    end
end

%% FUNCTIONS

function data = ExtractFrames(data, nSuperFrames, frameLen)
% Extract frame data from superframes
%
% This function takes in the Verasonics 3D superframe data and extracts it
% in individual frames.
%
% Parameters:
%   data         - A 3D array containing superframe data, where each 
%                  superframe is stored along the 3rd dimension.
%   nSuperFrames - The number of superframes to process.
%   frameLen     - The length of each individual frame.
%
% Returns:
%   data - A 3D array containing the extracted frames, with frames stored
%          along the 3rd dimension.

% Define the number of frames per superframe
nFramesPerSf = 10;

% Calculate the total number of frames to extract
nFrames = size(data,3)*nSuperFrames;

% Initialize a new array to hold the extracted frames
newData = zeros(frameLen, size(data,2), nFrames);

% Counter
i = 1;

for sf = 1:nSuperFrames
    for f = 1:nFramesPerSf
        % Compute the current frame index
        frame = (sf-1)*10 + f;

        % Extract the frame data and store it in the new array
        newData(:,:,i) = data(((f-1)*frameLen+1):f*frameLen,:,sf);

        i = i+1;
    end
end

% Update the output with the extracted frames
data = newData;
end

function RF = seperateElements(data,frame,connectorArray)
% Stores the RF data as element data in a struct and cuts it off at
% maxGridpoints. This will adjust the length of the RF data such that it
% equals the length of the input layer in the NN.
%
% Parameters:
%   data           - A 3D array containing RF data. The dimensions are
%                    assumed to be (gridpoints, channels, frames).
%   frame          - An integer specifying the frame index to process.
%   connectorArray - A 1D array specifying the channel indices (connectors)
%                    to extract.
%
% Returns:
%   RF - A struct array containing the RF data for each channel.
%        Each struct has the following fields:
%          - V: The RF signal data as a double array.
%          - fs: The sampling frequency of the RF data (62.5 MHz).
%

% Define the maximum number of grid points to process
maxGridpoints = 8446;

% Determine the number of elements (channels) to process
nElements = length(connectorArray);

% Create RF struct
RF = struct();

for el = 1:nElements
    % Get the channel index from the connector array
    line = connectorArray(el);

    % Extract RF signal and convert to double
    RF(el).V = double(data(1:maxGridpoints,line,frame)'); 

    % Assign sampling frequency
    RF(el).fs = 62.5e6;
end

end

function RF = scaleAndFilterRF(RF, TF_gain, TGC_gain, N, Hfilt, ROI1, ROI2, I, noise)
% Scales the RF lines by correcting for the transfer function differences
% between simulations and experiments, and compensates for the differences
% in gain used.
%
% Parameters:
%   RF        - Struct array containing RF data.
%   TF_gain   - Transfer function gain correction factor (scalar or array).
%   TGC_gain  - Time-gain compensation factor (scalar).
%   N         - Number of points for FFT computation (zero-padding length).
%   Hfilt     - Transfer function filter (frequency domain).
%   ROI1      - Start of the region of interest (ROI) in samples.
%   ROI2      - End of the region of interest (ROI) in samples.
%   I         - Transition width (in samples) for scaling in and out of the ROI.
%   noise     - Boolean flag to indicate whether to apply ROI scaling (false = apply).
%
% Returns:
%   RF - Struct array with processed RF data. The field `V` contains the
%        scaled and filtered time-domain RF signals.

% Number of elements (RF lines)
N_el = length(RF);

% Number of samples in each RF line
M = length(RF(1).V);

% Transfer function gain adjustment for each element
TF_gain = ones(1, N) * TF_gain;

% Define the ROI vector
G                   = ones(1,M);
G(1:(ROI1-I))       = 0;
G((ROI1-I+1):ROI1)  = linspace(1/I,1,I)./G((ROI1-I+1):ROI1); % Ramp up
G(ROI2+1:(ROI2+I))  = linspace(1,1/I,I)./G(ROI2+1:(ROI2+I)); % Ramp down
G(ROI2+I+1:M)       = 0;

for el = 1:N_el

    % Convert to frequency domain
    f_RF        = fft(RF(el).V, N);

    % Scale with the transfer function of the transducer
    f_RF_sc     = f_RF .* TF_gain;

    % Remove frequencies using transfer function Hfilt
    f_RF_sc_filt= f_RF_sc .* Hfilt;

    % Convert back to time domain
    RF_sc_filt       = real(ifft(f_RF_sc_filt, N));
    RF_sc_filt       = RF_sc_filt(1:M);

    % Scale with gains with ROI vector if the data is not a noise measurement
    if noise == false
        RF_sc_filt = RF_sc_filt .* G;       
    end

    % Apply time-gain compensation scaling
    RF(el).V    = RF_sc_filt * TGC_gain;    
end

end
