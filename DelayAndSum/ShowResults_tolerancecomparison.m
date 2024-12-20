% ShowResults_tolerancecomparison.m
% Compares images generated with different tolerance inputs for RF lines 
% generated with the Ref and SDC pulse.
%
% - No tol: Raw outputs of the networks
% - tol 1:  Network outputs but thresholded based on the optimal threshold
%           for a tolerance of 1
% - tol 4:  Network outputs but thresholded based on the optimal threshold
%           for a tolerance of 4
% Author:   Rienk Zorgdrager, University of Twente, 2024

%% Clear and close all
close all;
clc;
clear;

%% Settings
% Delimiter
delim = '\\';

% Image type (dl or sr)
super_resolved = true;
showbubbles = true;

% Define image segments for the plot
x3_1 = 0.04*1e3;
x3_2 = 0.06*1e3;
y3_1 = -0.01*1e3;
y3_2 = 0.01*1e3;

% Define variables of interest
tolerances = [1;4];             % Evaluated tolerace (grid points)
pulses = ["Ref", "SDC"];        % Evaluated pulses
filename = "RFDATA00001.mat";   % Evaluated file

%% PATHS CORRESPONDING TO DATASETS
paths = ["D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D\pulseSingle_Reference_OneCycle_500bubbles";
    "D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D\pulseChirp_Short_Downsweep_500bubbles"];

%% INITIALIZE THE FIGURES
f = figure('Position',[100,100,1400,1000]);
t = tiledlayout(3,2);

for p = 1:length(paths)
    % Load the raw file
    load(paths(p) + delim + filename)

    %% IMAGE WITHOUT TOLERANCE
    % Load the file
    filename_sr = strrep(filename, ".mat", "DAS_sr.mat");
    load(paths(p)+delim+"Images"+ delim + filename_sr)
    
    % Demodulate image and compute intensities
    img_demod = abs(img);
    img_demod2 = img_demod;
    mask = (x < 0.015) .* (x > -0.015);
    mask = repmat(mask',1,size(img_demod2,2));
    img_demod2(~mask) = 0;
    img_demod2(img_demod2<0.001)=nan;
    
    % Store intensities
    intensities(1+(p-1)).values = img_demod2;
    intensities(1+(p-1)).num_bub = 500;

    % Compute constrast colorbar limits
    pd = fitdist(img_demod2(:),'normal');
    disp('Intensity of noise free image')
    disp(pd)
    clim_lower_sr = pd.mu + 1*pd.sigma;
    clim_upper_sr = pd.mu + 8*pd.sigma;

    nexttile(1+(p-1));
    
    % Show the image
    show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr)
    title(sprintf('%s no tol', pulses(p)))
    xlim([x3_1 x3_2])
    ylim([y3_1 y3_2])
    set(gca, 'CLim', [clim_lower_sr, clim_upper_sr]);

    %% IMAGE WITH tol = 1 
    % Load the file
    filename_sr1 = strrep(filename, ".mat", "DAS_sr_tol1.mat");
    load(paths(p)+delim+"Images"+ delim + filename_sr1)
    
    % Demodulate
    img_demod = abs(img);
    img_demod2 = img_demod;
    img_demod2(~mask) = 0;
    img_demod2(img_demod2<0.001)=nan;
    img_demod_tol1 = img_demod;

    % Compute constrast
    pd = fitdist(img_demod2(:),'normal');
    disp('Intensity of noise free image')
    disp(pd)
    clim_lower_sr = pd.mu + 1*pd.sigma;
    clim_upper_sr = pd.mu + 8*pd.sigma;

    % Store intensities
    intensities(3+(p-1)).values = img_demod2;
    intensities(3+(p-1)).num_bub = 500;

    nexttile(3+(p-1));

    % Show the image
    show_reconstruction(z,x,img_demod_tol1,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr)
    title(sprintf('%s tol = 1', pulses(p)))
    xlim([x3_1 x3_2])
    ylim([y3_1 y3_2])
    set(gca, 'CLim', [clim_lower_sr, clim_upper_sr]);

    %% IMAGE WITH tol = 4
    % Load the datafiles
    filename_sr4 = strrep(filename, ".mat", "DAS_sr_tol4.mat");
    load(paths(p)+delim+"Images"+ delim + filename_sr4)

    % Demodulate the iamge
    img_demod = abs(img);
    img_demod2 = img_demod;
    img_demod2(~mask) = 0;
    img_demod2(img_demod2<0.001)=nan;
    img_demod_tol4 = img_demod;

    % Store the intensities
    intensities(5+(p-1)).values = img_demod2;
    intensities(5+(p-1)).num_bub = 500;

    % Compute constrast
    pd = fitdist(img_demod2(:),'normal');
    disp('Intensity of noise free image')
    disp(pd)
    clim_lower_sr = pd.mu + 1*pd.sigma;
    clim_upper_sr = pd.mu + 8*pd.sigma;
    
    % Show the image
    nexttile(5+(p-1));
    show_reconstruction(z,x,img_demod_tol4,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr)
    title(sprintf('%s tol = 4', pulses(p)))
    xlim([x3_1 x3_2])
    ylim([y3_1 y3_2])
    set(gca, 'CLim', [clim_lower_sr, clim_upper_sr]);
end

%% Intensity histograms
% Show histogram of the intensity values in the super-resolved image
histFig = figure('units','inch','position',[0,0,3.5,4]);
h = tiledlayout(histFig,2,3);

% Plot the histograms for every figure in a subplot.
for d = 1:length(intensities)     % Reopen figure
    nexttile
    histogram(intensities(d).values);
    title(intensities(d).num_bub)
    grid on
    xlabel('intensity (a.u.)')
    ylabel('counts')
    xlim([0,1])
    set(gca,'FontSize',12)
end
