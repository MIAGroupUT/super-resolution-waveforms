% Delay-and-sum reconstruction of the bubble image from the element data
% simulated with RF_simulatorFINAL. After super-resolving the RF data.
%
% Nathan Blanken, University of Twente, 2021
% Adjusted by Rienk Zorgdrager, University of Twente, 2024

%% Clear workspace and command window
clear
clc

%% SETTINGS
delim = '\';

%% INPUTS
% Enter the noise level with which the data is saturated.
% For images generated without noise, enter: additional_specification = "";
% For images generated with networks on a fixed noise level, enter: additional_specification = "_noise#";
% For images generated with networks on a noise range, enter:additional_specification = "_noiserange#-#";
% For experimental images, enter: additional_specification = "_absolutenoise0.752713";

additional_specification = "_absolutenoise0.752713"; 

% Enter user-defined tolerance
tolerance = 4 ;

% Define whether the data is experimental
experimental_data = true;

% Define whether the data processing is done to make a movie
make_movie = true; % Set to true if you wish to process ALL RF files

% Define whether the entire image needs to be processed.
entire_image = false;

% Define which figure shall be made
plotmode = "bubbles"; % Sections used for two different figures, enter 'pulses' for PulseComparison, enter 'nBubbles' for nBubblesComparison

%% FIGURE SETTINGS
dpi = 600;

%% PATHS
folderpath = uigetdir('D:\SRML-1D-pulse-types\Results\', "Select the folder containing the 2D .mat files with RF data");
directory = dir(folderpath);
directory = directory(3:end);

%% FILTER THE DIRECTORIES IF NEEDED
if contains(additional_specification,"_noiserange") || contains(additional_specification,"_absolutenoise")
    disp("Evaluating " + additional_specification)
elseif plotmode == "pulses" % Select the models to be evaluated for noise
    filt1 = contains({directory.name},'pulseSingle_Reference_OneCycle_500bubbles');
    filt2 = contains({directory.name},'pulseSingle_Short_MedF_500bubbles');
    filt3 = contains({directory.name},'pulseSingle_Long_MedF_500bubbles');
    filt4 = contains({directory.name},'pulseChirp_Long_Downsweep_500bubbles');
    filt5 = contains({directory.name},'pulseChirp_Short_Downsweep_500bubbles');
    combinedFilter = boolean(filt1+filt2+filt3+filt4+filt5);
    directory = directory(combinedFilter);
end

% Load environmental settings and transducer characteristics if needed
if experimental_data == true
    addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\Experimental_validation\extra_files')
    load("env_settings.mat")
    load("extra_files\Trans.mat");                  % Receiver characteristics
end

%% CONSTRUCT BUBBLE IMAGES

for j=1:length(directory)

    % Define directories
    dirpath = string(directory(j).folder) + string(delim) + string(directory(j).name);

    % Get the first RF data files
    files = dir(dirpath);
    if make_movie == false
        RF_ind = find(contains({files.name},'RFDATA')==1,1,'first');
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

    % Domain and transducer properties
    width = domain.width;       % domain width (m)
    depth = domain.depth;       % domain depth (m)
    N = length(RF(1).V);        % number of samples per RF line
    Nelem = length(RF);         % number of transducer elements
    Fs = RF(1).fs;              % sample rate (Hz)
    t = (0:(N-1))/Fs;           % time axis (s)

    x_el = linspace(-width/2,width/2,Nelem); % Element positions (m)

    %% WAVE PROPERTIES
    c = liquid.c;              % speed of sound in the medium (m/s)
    f0 = 2.5e6;                % Center freq. of P4-1 (Hz)
    lambda = c/f0;             % wavelength (m)

    focus  = inf;              % Plane wave imaging

    %% IMAGE SPECIFICATIONS
    % Dimensions of the reconstructed image:
    IM_WI = width*1.5;          % Width of the reconstructed image (m)
    IM_DE = depth;              % Depth of the reconstructed image (m)

    % Define pixel size for image reconstruction (m)
    if make_movie == true || entire_image == true
        pix_siz = 2e-5;
    else
        pix_siz = 1e-5;
    end

    % Reconstruction boundaries (m)
    if entire_image == false
        if experimental_data == true
            x_rec1 = -0.01;
            x_rec2 = -x_rec1;
            z_rec1 = 0.040;
            z_rec2 = 0.060;
        else
            if plotmode == "bubbles"
                x_rec1 = 0;
                x_rec2 = 1e-2;
                z_rec1 = 45e-3;
                z_rec2 = 55e-3;
            elseif plotmode == "pulses"
                x_rec1 = -5e-3;
                x_rec2 = 5e-3;
                z_rec1 = 40e-3;
                z_rec2 = 50e-3;
            end
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

    % Find duration of pulse in lens (samples)
    if experimental_data == true
        f               = Trans.frequency*1e6;      % Transducer center frequency (Hz) 
        dt_lens_corr    = 2*Trans.lensCorrection/f; 
        lens_dur        = dt_lens_corr*RF(1).fs;    % Travel time in lens (samples)
    else
        lens_dur        = 0;
    end

    sig_dur     = 1;    % Distance to peak intensity. For SR imaging, this is 1 sample.

    % Total duration for which needs to be corrected (samples)
    tot_dur     = lens_dur + sig_dur;

    %% COMPUTE ANGLE SENSITIVITY FUNCTION
    
    sens = ones(1,101); % After resolving, we do not need to correct for angle sensitivity
    
    % Define angles corresponding to the values in sens
    angles  = (-pi/2:pi/(length(sens)-1):pi/2);     
    
    % Get a fit through the angle sensitivity
    [interp_fun, ~] = createinterp(angles, sens);

    %% Compute DAS matrix
    % Compute the DAS matrix which defines the segments in the RF signal to
    % include in each pixel
    [M_DAS, apod] = compute_das_matrix(t, x_rec, z_rec, x_el, c, Fs, focus, tot_dur, interp_fun);

    for ind = 1:length(RF_ind)

        % Close all figures
        close all

        % Indices of file names
        idx = RF_ind(ind);
        filename = files(idx).name;

        filepath =  dirpath + string(delim) + filename;

        % Load the metadata from the original MATLAB files:
        load(filepath)

        % Load the super-resolved RF data:
        if exist('tolerance') == 1
            filepath_sr = dirpath + delim + "sr_data" + delim + strrep(filename, '.mat', "_sr_tol" + string(tolerance) + additional_specification+".txt");
        else
            filepath_sr = dirpath + delim + "sr_data" + delim + strrep(filename, '.mat', "_sr"+ additional_specification+".txt");
        end

        RF_matrix   = read_rf_txt(filepath_sr,Nelem);
        RF_matrix   = RF_matrix';

        %% TIME GAIN COMPENSATION (TGC) (NOT USED IN SR IMAGING)
        RF_TGC = RF_matrix;

        %% DELAY-SUM RECONSTRUCTION

        tic
        img = delay_and_sum_matrix(RF_TGC, M_DAS, apod, Nx_rec, Nz_rec);
        toc

        if make_movie == false
            % Store a pdf image
            figure('Visible','off');
            img_sr = imshow(img);

            img_path = dirpath+delim+directory(j).name+"DAS_sr"+ additional_specification+".pdf";
            exportgraphics(gca, img_path, 'ContentType', 'vector', 'Resolution', dpi)
        end

        %% SAVE THE RESULTS
        x = -IM_WI/2:pix_siz:IM_WI/2;       % Lateral coordinates (m)
        z = 0:pix_siz:IM_DE;                % Axial coordinates (m)

        if entire_image == false
            fname = string(filename(1:end-4)) + "_section_";
            if experimental_data == true
                % filename is ok
            elseif plotmode == "bubbles"
                fname = fname + "bubbles_";
            elseif plotmode == "pulses"
                fname = fname + "pulses_";
            end
        else 
            fname = string(filename(1:end-4));
        end
        
        if exist('tolerance') == 1
            % savepath = dirpath+delim+'Images'+ delim + fname + "DAS_sr" + "_tol" + string(tolerance) + additional_specification+".mat";
            savepath = dirpath+delim+'Images'+ delim + fname + "for_recon_DAS_sr" + "_tol" + string(tolerance) + additional_specification+".mat";
        else
            savepath = dirpath+delim+'Images'+ delim +fname + "DAS_sr" + additional_specification+".mat";
        end
        save(savepath,'img','x','z','x_rec','z_rec')
    end

end

