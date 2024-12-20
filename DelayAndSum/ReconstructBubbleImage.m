% Delay-and-sum reconstruction of the bubble image from the element data
% simulated with RF_simulatorFINAL.
%
% Nathan Blanken, University of Twente, 2021
% Adjusted by Rienk Zorgdrager, University of Twente, 2024

%% Clear workspace and command window
clear
clc

%% SETTINGS
delim = '\';
alpha = 0.03; % attenuation coefficient used for TGC (dB/cm/MHz), only used for experimental data

%% INPUTS

% Define whether the data processing is done to make a movie
make_movie = true; % Set to true if you wish to process ALL RF files

% Define whether the entire image needs to be processed.
entire_image = false;

%% ADD MODULES
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\RF_simulator\functions');
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\RF_simulator\extra_files');
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\RF_simulator\microbubble-simulator\functions')
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\Experimental_validation\extra_files')

%% LOAD DATA AND METADATA
Hfit = load('ReceiveTransferFunctionFit.mat','Hfit');   % Receiver transfer function
load("extra_files\Trans.mat");                  % Receiver characteristics, these are specific for the pulses used during these measurements

%% PATHS TO FILES

folderpath = uigetdir('D:\SRML-1D-pulse-types\Results', "Select the folder containing the 2D .mat files with RF data");

% Check if the data is experimental
if contains(folderpath,"Experiments")
    experimental_data = true;
    load("env_settings.mat")
else
    experimental_data = false;
end

% Filter the directories
directory = dir(folderpath);
directory = directory(3:end);
directory = directory(~strcmp({directory.name}, 'Images')); % Ignore the images folder

pulse_name = string();

%% CONSTRUCT BUBBLE IMAGES

for j=1:length(directory)

    % Define directories
    dirpath = string(directory(j).folder) + string(delim) + string(directory(j).name);

    % Get the first RF data files
    files = dir(dirpath);
    if make_movie == false
        % RF_ind = find(contains({files.name},'RFDATA')==1,1,'first');

        % We show the 11th image
        RF_ind = 11;
    else
        RF_ind = find(contains({files.name},'RFDATA')==1);
    end

    % Load the first file to get the right properties
    ind = 1;
    idx = RF_ind(ind);
    fname = files(idx).name;

    filepath =  dirpath + string(delim) + fname;

    % Load the metadata from the original MATLAB files:
    load(filepath)

    % Get the pulse name
    pulse_name_new = split(dirpath,'\');
    pulse_name_new = split(pulse_name_new(end),'_');
    
    % Check if the folder name is correct
    if length(pulse_name_new) > 3   % Folder belongs to synthetic bubble dataset
        pulse_name_new = join(pulse_name_new(1:end-1),'_');
    else                            % Folder name is the pulse name
        pulse_name_new = join(pulse_name_new(1:end),'_');
    end
    
    % If the dataset is constructed with the same pulse, we can use the
    % same settings. If not, we need to recompute the M_DAS and apod 
    % matrices.
    if pulse_name_new ~= pulse_name
        % Domain and transducer properties
        width = domain.width;       % domain width (m)
        depth = domain.depth;       % domain depth (m)
        N = length(RF(1).V);        % number of samples per RF line
        Nelem = length(RF);         % number of transducer elements
        Fs = RF(1).fs;              % sample rate (Hz)
        t = (0:(N-1))/Fs;           % time axis (s)

        x_el = linspace(-width/2,width/2,Nelem); % Element positions (m)

        %% WAVE PROPERTIES
        c = liquid.c;               % speed of sound in the medium (m/s)
        f0 = pulse.f;               % centre frequency (Hz)
        lambda = c/f0;              % wavelength (m)

        focus  = inf;               % Plane wave imaging

        %% IMAGE SPECIFICATIONS
        % Dimensions of the reconstructed image:
        IM_WI = width*1.5    ;          % Width of the reconstructed image (m)
        IM_DE = depth    ;              % Depth of the reconstructed image (m)

        % Define pixel size for image reconstruction (m)
        if make_movie == true || entire_image == true
            pix_siz = 2e-5;
        else
            pix_siz = 1e-5;
        end

        % Reconstruction boundaries (m)
        if entire_image == false
            if experimental_data == false
                x_rec1 = 0;
                x_rec2 = 1e-2;
                z_rec1 = 45e-3;
                z_rec2 = 55e-3;
            else
                x_rec1 = -0.01;
                x_rec2 = -x_rec1;
                z_rec1 = 0.040;
                z_rec2 = 0.060;
            end
        else
            x_rec1 = -IM_WI/2;
            x_rec2 = IM_WI/2;
            z_rec1 = 0;
            z_rec2 = IM_DE;
        end

        % Pixel coordinates
        x_rec   = x_rec1:pix_siz:x_rec2;    % x coordinates
        z_rec   = z_rec1:pix_siz:z_rec2;    % z coordinates

        % Reconstruction pixels
        Nx_rec = length(x_rec);         % Number of pixels
        Nz_rec = length(z_rec);         % Number of pixels

        % Compute the approximate duration of the pulse in number of samples:
        Ndel = 0;
        
        % Find two-way pulse
        if experimental_data
            two_way_pulse = pulse.Wvfm2Wy;
        else
            two_way_pulse = receiveTransferFunction(pulse.p_norm, Hfit.Hfit, pulse.fs);
        end
        
        % Find maximum of pressure pulse (samples)
        if contains(directory(j).name, "Chirp", 'IgnoreCase', true)
            I = 0;
        else
            [~, I] = max(abs(hilbert(two_way_pulse)));
        end

        % Find duration of pulse in lens (samples)
        if experimental_data == true
            f               = Trans.frequency*1e6;
            dt_lens_corr    = 2*Trans.lensCorrection/f;
            lens_dur        = dt_lens_corr*Fs;
        else
            lens_dur        = 0;
        end

        % Find pulse duration (samples)
        sig_dur     = (I + Ndel)/pulse.fs*Fs ;

        % Total duration for which needs to be corrected (samples)
        tot_dur     = sig_dur + lens_dur;

        %% Compute angle sensitivity function
        if experimental_data == true
            sens = Trans.ElementSens;
            angle_correction = true;
        else
            sens = ones(1,101);
            angle_correction = false;
        end

        angles  = (-pi/2:pi/(length(sens)-1):pi/2);     % Angles corresponding to the values in sens

        % Get a fit through the angle sensitivity
        [interp_fun, ~] = createinterp(angles, sens);

        %% Compute DAS matrix
        % Compute the DAS matrix which defines the segments in the RF signal to
        % include in each pixel
        [M_DAS, apod] = compute_das_matrix(t, x_rec, z_rec, x_el, c, Fs, focus, tot_dur, interp_fun);

    end

    % Overwrite the pulse_name
    pulse_name = pulse_name_new;

    % Loop over the datafiles in the folder
    for ind = 1:length(RF_ind)

        % Close all figures
        close all

        % Indices of file names
        idx = RF_ind(ind);
        fname = files(idx).name;

        disp(['Processing ' fname])

        filepath =  dirpath + string(delim) + fname;

        % Load the metadata from the original MATLAB files:
        load(filepath)

        % Compress the chirps
        if contains(directory(j).name,"Chirp") == true || contains(directory(j).name,"chirp") == true
            disp('Chirp is being compressed')
            RF = compressChirp(pulse, RF, Hfit);
        end

        % Convert RF struct to matrix
        RF_matrix = [RF.V];
        RF_matrix = reshape(RF_matrix, N, Nelem)';  % matrix of element RF data
        
        % Convert to double
        if class(RF_matrix) ~= "double"
            RF_matrix = double(RF_matrix);
        end

        %% TIME GAIN COMPENSATION (TGC)
        % Apply a TGC to compensate for the 1/r decay of scattered pressure:

        if experimental_data == true
            TGC = 10.^(alpha*f0*1e-6*c.*t.*1e2./20);
        else
            TGC = t;
        end
        
        TGC = repmat(TGC,size(RF_matrix,1),1);
        
        % Compute the new RF signal
        RF_TGC = RF_matrix.*TGC;

        %% APPLY LOW FREQUENCY FILTERING (exp. data only)
        % Compensate for the acquisition errors in the experimental data

        if experimental_data == true

            % Preallocate filtered array
            RF_data_filt = zeros(size(RF_TGC));

            for ifr  = 1:size(RF_TGC,3)

                % Compute local RF line (for image sequences)
                RF_loc = RF_TGC(:,:,ifr);

                % Fourier transform of the RF lines
                TF = fft(RF_loc');
                fr = linspace(0,Fs,size(RF_loc,2)); % frequency vector

                % Remove frequencies below 10kHz
                FR_CUT = find(fr<1e5,1,'last');
                TF(1:FR_CUT,:) = 0;
                TF(floor(length(fr)/2):end,:) = 0;

                % Inverse Fourier transform
                RF_loc2 = 2.*real(ifft(TF));
                RF_data_filt(:,:,ifr) = RF_loc2';

            end
        else
            RF_data_filt = RF_TGC;
        end

        %% HILBERT TRANSFORM
        RF_HB = hilbert(RF_data_filt.').';

        clear RF_TGC

        % Delay and sum reconstruction:
        tic
        img = delay_and_sum_matrix(RF_HB, M_DAS, apod, Nx_rec, Nz_rec);
        toc

        %% CONVERT TO REAL NUMBERS
        img = abs(img);

        %% LOG COMPRESSION
        % Log compression:
        img_max = max(img,[],[1,2]); % Maximum value (1-by-1-by-Nframes)
        img_log = 20*log10(img./img_max);

        img_log(~isfinite(img_log)) = nan;

        %% SAVE THE RESULTS
        % Pixel coordinates
        x = -IM_WI/2:pix_siz:IM_WI/2;       % Lateral coordinates (m)
        z = 0:pix_siz:IM_DE;                % Axial coordinates (m)
        
        % Define the filename
        fname = erase(fname,".mat");

        if entire_image == false
            fname = string(fname) + "_section_";
        end

        savedir = dirpath+delim+"Images";

        if ~exist(savedir) == 1
            mkdir(savedir)
        end

        % Save the data
        save(savedir+delim+fname+'for_recon_DAS_dl.mat','img_log','x','z','x_rec','z_rec')

        %% Show a middle section of the image
        if entire_image == true
            % Limits
            x1 = -0.01;
            x2 = -x1;
            z1 = 0.045;
            z2 = 0.065;

            lat_range   = (x>x1 & x<x2);
            ax_range    = (z>z1 & z<z2);

            % Segment the middle of the image
            section = img_log(lat_range,ax_range);

            % Plot the full image
            figure('Visible', 'Off');
            imshow(img_log)
            fig = gca;
            set(fig, 'Visible', 'Off')
            
            % Labels and axis
            xticklabels();
            clim([min(min(section)) max(max(section))]);
            title(directory(j).name)

            % Plot a section of the image
            figure('Visible', 'Off');
            imshow(section)
            fig = gca;
            set(fig, 'Visible', 'Off')
            clim([min(min(section)) max(max(section))])
            colorbar
            title(directory(j).name)
        end
    end
end

function RF = compressChirp(pulse, RF, Hfit)

rescaling = true; % Rescale to original pulse amplitude

p_norm = pulse.p_norm;
downsample = pulse.fs / RF(1).fs;

% Compress chirps
% Convolve pressure wave with receiver transfer function:
V_pulse = receiveTransferFunction(p_norm, Hfit.Hfit, pulse.fs);
V_pulse = V_pulse/max(abs(V_pulse));    % Normalise expected pulse
V_pulse = V_pulse(1:downsample:end);    % Downsample the expected pulse

% Define matched filter
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

% Apply the filter in the RF line
for el = 1:length(RF)
        RF(el).V = conv(RF(el).V,matched_filter)/A;
    RF(el).V(1:(N-1)) = []; % Cut off the zero-padded start of convolution
end

end