%% Make a video of the experimental acquisitions
% This code writes a video from the beamformed images captured with the
% Verasonics. It asks to select the directory of images acquired with the 
% same pulse (this which is divided in process_exp_results.m). Consequently
% It displays the diffraction-limited (DL) image on the leftand the 
% super-resolved (SR) image on the right. The video is stored as an
% .mp4 file.
%
% Rienk Zorgdrager, University of Twente, 2024

%% CLEAR AND CLOSE ALL
clear
close all

%% INPUTS
% Delimiter
delim = '\';

% User-defined tolerance
tolerance = 4;

% Figure and axes size
figW = 5;           % Figure width in inches
figH = 7;           % Figure height in inches
axW = 2.5;          % Axis width in inches
axH = 2.5;          % Axis height in inches
spacingMiddle = 1;  % Space between the axis in inches
spacingSides = (figW - 2*axW - spacingMiddle)/2;  % Space next to the axes
fontSize = 6;       % Font size

% Name of the video
video_name = 'video_DL_vs_SR';

if exist("tolerance") == 1
    video_name = video_name+"_tol"+string(tolerance);
end

%% OTHER SETTINGS
% Acquisition settings
interAcqTime = 500;                 % time in microseconds between acquisitions
interFrameTime = 2*interAcqTime;    % time in microseconds between two frames with the same pulse

resScaling = 5; % Increase size of figures, text and labels so that the resolution is higher

% Define image section
x1 = -0.01;
x2 = -x1;
z1 = 0.040;%0.045;
z2 = 0.060;%0.065;

% Colorbar limits
c1 = -40; %dB
c2 = 0; %dB

% No ground thruth location available.
show_bubbles = false;
bubbles = [];

%% ADD PATH TO DAS FOLDER
addpath('C:\Users\rienk\OneDrive - University of Twente\Documents\Projects\US pulse shapes for deep learning applications\SRML-1D-pulse-types\DelayAndSum')

%% SELECT THE DATAFILES
% Select the folder in which the files are stored
folderpath = uigetdir('D:\SRML-1D-pulse-types\Results\Experiments\Data\Processed');

if contains(folderpath,'pulse') == false
    error('Please select a datafile containing data from one pulse only')
end

img_folder = strcat(folderpath,delim,"Images");
directory = dir(img_folder);
directory = directory(3:end); % Remove the dots

% Select only the mat files of interest
if exist("tolerance") == 1
    filt_sr = contains({directory.name},'_section_DAS_sr_tol'+string(tolerance));
else
    filt_sr = contains({directory.name},'_section_DAS_sr');
end
filt_dl = contains({directory.name},'_section_DAS_dl');

% Obtain directories for sr and dl images
directory_sr = directory(filt_sr);
directory_dl = directory(filt_dl);

% Get the pulse name
subfolders = split(directory(1).folder,'\');
pulse_name = subfolders{7}; % name of the pulse
if contains(directory(1).folder,"chirp") || contains(directory(1).folder,"Chirp")
    pulse_name = "Upsweep chirp";
elseif contains(directory(1).folder,"SIP")
    pulse_name = "Short imaging pulse";
end

%% INITIALIZE THE VIDEOWRITER
v = VideoWriter(video_name,'MPEG-4');
v.FrameRate = 10;
v.Quality = 100;
open(v)

for i = 1:length(directory_sr)

    % Compute the time
    t = i*interFrameTime*1e-3; % In milliseconds

    % Define file names
    filename_sr = directory_sr(i).name;
    filename_dl = directory_dl(i).name;
    
    % Define file paths
    filepath_sr = strcat(img_folder,delim,filename_sr);
    filepath_dl = strcat(img_folder,delim,filename_dl);

    %% LOAD THE FILES
    % Diffraction-limited
    load(filepath_dl); % The loaded image is the matrix 'img_log'
    x_dl = x;
    z_dl = z;

    % Super-resolved
    load(filepath_sr); % The loaded image is the matrix 'img'

    %% DEFINE COLORBAR LIMITS SEGMENTED IMAGE
    clim_low_dl = c1;   % Lower limit
    clim_up_dl  = c2;   % Upper limit

    % Fit probability density function to define contrast limits
    if i == 1
        pd = fitdist(img(:),'normal');
        % pd = fitdist(section_sr(:),'normal');
        clim_low_sr = pd.mu + 1*pd.sigma;
        clim_up_sr  = pd.mu + 8*pd.sigma;
    end

    %% MAKE RECONSTRUCTED IMAGE
    % Initialize the figure
    f = figure('Units', 'inches', 'Position', [2 2 figW*resScaling figH*resScaling],"Visible","off");

    % Use tiledlayout
    tiles = tiledlayout(1,2, TileSpacing="compact");

    % Make the diffraction-limited image
    ax1 = nexttile;
    show_reconstruction(z_rec,x_rec,img_log,false,bubbles,show_bubbles,clim_low_dl,clim_up_dl);
    tex1 = text(z2*1e3-5, x2*1e3-1, {string(t)+' ms '}, 'Color', 'w', 'FontName','Times New Roman');
    tex2 = text(z1*1e3+1, x2*1e3-1, filename_dl(1:11), 'Color', 'w', 'FontName','Times New Roman');

    % Make the super-resolved image
    ax2 = nexttile;
    show_reconstruction(z_rec,x_rec,img,true,bubbles,show_bubbles,clim_low_sr,clim_up_sr);
    tex3 = text(((z2-z1)*0.8+z1)*1e3, x2*1e3-1,"tol = "+string(tolerance),'Color','k', 'FontName','Times New Roman');
    
    % Formatting
    ax1.Colormap = flipud(ax1.Colormap);    % Invert the colormap of the DL image
    sgtitle(pulse_name,'fontweight','bold')
    f.Color = 'w';
    hold on
    fontsize(gcf,scale=resScaling/1.5)

    %% WRITE FRAME TO VIDEO
    % Get the frame
    frame = getframe(gcf);

    % Write to video
    writeVideo(v,frame)

end

% Close videoWriter object
close(v)

% Move file to corresponding folder
movefile(video_name+".mp4",folderpath) 