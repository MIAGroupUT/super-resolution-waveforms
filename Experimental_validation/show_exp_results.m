%% Display generate experimental images 
% Script to display the DAS images from the experimental data. This is
% Fig. 10 in the manuscript. This script generates two images. The first
% one is an svg file with the axes, the second is a pdf file with exactly
% the same data, but with a higher image quality. The two are merged in
% a vector graphics editor program.
%
% Rienk Zorgdrager, University of Twente, 2024

%% CLEAR AND CLOSE ALL
clear
close all

%% INPUTS
delim = '\\';

% Noise specification
noise_specification = "_absolutenoise0.752713";

% Image specifications
figW = 7.16;    % Figure width in inches
figH = 6;       % Figure height in inches

% Specifications of the cross-sections
cross_secs = [-0.0032, .053, .058]; 

% Filenames
frame = 15;     % Frame number of the SIP images to be plotted.
filenames = {};
filenames{1} = sprintf('RFDATA%05d',frame);     % Filename for the SIP
filenames{2} = sprintf('RFDATA%05d',frame+1);   % Filename for the chirp

% Network output specifications
tolerance = 4;

if exist("tolerance") == 1
    additional_specification = "_tol" + string(tolerance) + noise_specification;
else
    additional_specification = noise_specification;
end

%% SETTINGS

% No ground thruth location available.
show_bubbles = false;
bubbles = [];

% Define image section
x1 = -0.01;
x2 = -x1;
z1 = 0.040;
z2 = z1+0.02;

% Colorbar limits
c1 = -40;   %dB
c2 = 0;     %dB

% Add paths to export_fig and DelayAndSum module
addpath('C:\Users\rienk\OneDrive\Bureaublad\export_fig-master')
addpath("C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\DelayAndSum")

%% DIRECTORIES
% Select the folder with the processed acquisition data.
parent_path = uigetdir("D:\SRML-1D-pulse-types\Results\Experiments\Data\Processed","Select the processed data acquisition to be shown");

% Obtain directories of the pulses
pulse_dir = dir(parent_path);
pulse_dir = pulse_dir(~ismember({pulse_dir.name},{'.','..'}));

fprintf('Files analyzed: \n Short Imaging Pulse: %s \n Chirp: %s \n', filenames{1}, filenames{2})

%% INITIALIZE THE FIGURE

f = figure('Units', 'inches', 'Position', [2 2 figW figH],"Visible","on");

% Use tiledlayout
tiledlayout(3,2, TileSpacing="compact")

for p = 1:length(pulse_dir)
    
    % Get the pulse name and the directory to the images
    pulse = pulse_dir(p).name;
    img_dir = strcat(pulse_dir(p).folder,delim,pulse,delim,"Images");

    files = dir(img_dir);

    if length(files)<3
        disp(['No files found in image folder for pulse ', pulse])
        return
    end
    
    % Find all .mat files
    files = files(endsWith({files.name},'.mat'));

    % Paths to dl and sr files
    filepath_dl = strcat(img_dir,delim,filenames(p),'_section_for_recon_DAS_dl');
    filepath_sr = strcat(img_dir,delim,filenames(p),'_section_for_recon_DAS_sr', additional_specification);

    %% LOAD THE RAW FILES
    load(filepath_dl)
    load(filepath_sr)

    % Demodulate log compressed image
    if exist('img_log') ~= 1
        img_demod = demodulateImage(img_log);
    else
        img_demod = 10.^(img_log / 20);
    end

    %% SEGMENT IMAGE
    % Define image segment
    widthratio  = 0.7;
    x_upper = widthratio * max(x_rec);
    x_lower = widthratio * min(x_rec);

    %% CONTRAST VALUES
    % Find indices
    [~,I] = min(abs(cross_secs(1)-x_rec));
    [~,J1] = min(abs(cross_secs(2)-z_rec));
    [~,J2] = min(abs(cross_secs(3)-z_rec));
    
    contrast_dl = img_demod(I,J1:J2); % Contrast value for the DL image
    contrast_dl = (contrast_dl-min(contrast_dl))/(max(contrast_dl)-min(contrast_dl)); % Normalize the score
    contrast_sr = img(I,J1:J2); % Contrast value for the SR image
    contrast_sr = (contrast_sr-min(contrast_sr))/(max(contrast_sr)-min(contrast_sr)); % Normalize the score

    %% DEFINE COLORBAR LIMITS SEGMENTED IMAGE
    clim_low_dl = c1;   % Lower limit
    clim_up_dl  = c2;   % Upper limit

    % Get the presented image section
    x_sec = (x_rec < x_upper & x_rec > x_lower);
    img_sec = img(:,x_sec);

    % Fit probability density function to define contrast limits
    pd = fitdist(img_sec(:),'normal');
    clim_low_sr = pd.mu + 1*pd.sigma;
    clim_up_sr  = pd.mu + 8*pd.sigma;

    %% MAKE RECONSTRUCTED IMAGE
    % Show the diffraction-limited and super-resolved image. Also, display
    % a cross-section, indicated with a dashed line. Plot the contrast
    % along this line in the bottom tiles.

    % Diffraction-limited
    ax_dl(p) = nexttile(1*(p-1)+1);
    show_reconstruction(z_rec,x_rec,img_log,false,bubbles,show_bubbles,clim_low_dl,clim_up_dl)
    hold on;
    plot([cross_secs(2)*1e3 cross_secs(3)*1e3], [cross_secs(1)*1e3 cross_secs(1)*1e3],"--", Color="#EDB120")
    rectangle('Position',[cross_secs(2)*1e3 (cross_secs(1)*1e3-1) (cross_secs(3)*1e3-cross_secs(2)*1e3) 2], 'EdgeColor','w')
    ylim([x_lower*1e3, x_upper*1e3])

    % Super-resolved
    ax_sr(p) = nexttile(1*(p-1)+3);
    show_reconstruction(z_rec,x_rec,img,true,bubbles,show_bubbles,clim_low_sr,clim_up_sr)
    hold on;
    plot([cross_secs(2)*1e3 cross_secs(3)*1e3],[cross_secs(1)*1e3 cross_secs(1)*1e3],"--", Color="b")
    rectangle('Position',[cross_secs(2)*1e3 (cross_secs(1)*1e3-1) (cross_secs(3)*1e3-cross_secs(2)*1e3) 2], 'EdgeColor','k')
    ylim([x_lower*1e3, x_upper*1e3])

    % Contrast along cross-section
    ax_contrast(p) = nexttile(1*(p-1)+5);
    hold on;
    box on;
    x_values = z_rec(J1:J2)*1e3;
    plot(x_values, contrast_dl)
    plot(x_values, contrast_sr)
    xlim([cross_secs(2)*1e3 cross_secs(3)*1e3])
    ylim([0,1.1])
    ax_contrast(p).PlotBoxAspectRatio = ax_dl(p).PlotBoxAspectRatio;
    
end

%% FIGURE FORMATTING
% Format the axes
set(ax_dl, 'Colormap', flipud(ax_dl(1).Colormap), 'FontName', 'Times New Roman')
set(ax_sr,'FontName', 'Times New Roman')
set(ax_dl, 'fontsize', 6)
set(ax_sr, 'fontsize', 6)
set(ax_contrast, 'fontsize', 6, 'FontName', 'Times New Roman')
xticks(ax_contrast,unique(round(x_values,0)))

% Figure background
f.Color = 'w';

%% EXPORT THE FIGURE
% Export the figure as svg
export_fig exp_reconstruction_frame -svg -painters
sourceFile = "exp_reconstruction_frame.svg";
destinationFile = "D:\SRML-1D-pulse-types\Results\Figures" + delim + sourceFile;
movefile(sourceFile, destinationFile)

% Export the high-quality images
set(0,'units', 'inches')
f.Position = get(0,'screensize');
export_fig exp_reconstruction_images.pdf -native
sourceFile = "exp_reconstruction_images.pdf";
destinationFile = "D:\SRML-1D-pulse-types\Results\Figures" + delim + sourceFile;
movefile(sourceFile, destinationFile)

%% FUNCTIONS
function img = demodulateImage(img)
% Demodulation of the signals
img_demod = zeros(size(img));

for k = 1:size(img_demod,1)
    img_demod(k,:) = abs(hilbert(img(k,:)));
end

% Log compression
maximg = max(max(img_demod));
img_log = 20*log10(img_demod/maximg);

end