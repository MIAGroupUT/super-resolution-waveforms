% Script to create the delay-and-sum figures in the manuscript. There are
% two main figures used in the manuscript:
% - nBubbleComparison:  Compares DL and SR images for different microbubble
%   densities
% - pulseComparison:    Compares SR images by different pulse excitations.

%% CLEAR AND CLOSE ALL
clear
close all

settings.delim = "\\";

%% INPUTS
tolerance = 4;

% Paths
settings.parent_path = "D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D";  % Path to datasets
settings.parent_savedir = "D:\SRML-1D-pulse-types\Results\Figures";              % Save directory for figures
datasets = dir(settings.parent_path);
datasets = datasets(3:end);

% Plot type
plotmode = "pulses";                  % Enter "nBubbles", "pulses" or "all"
additional_specification = "_noise128"; % Only used when plotmode = "pulses"

% Add path to export_fig module
addpath('C:\Users\rienk\OneDrive\Bureaublad\export_fig-master')

%% PLOT SETTINGS
% Figure settings
settings.dpi = 600;
settings.figWidth = 7.16; %Inches
settings.fontSize = 6;

% Specify cross-sections
settings.cross_secs = [
    1.64553,49,52;
    1.64375,50,53;
    1.93755,47,50;
    1.36694,47,50;]; % For densities of resp. 1000,100,2500,500. This is chosen such the cross section is at exactly one bubble.

settings.nBubblesList = [100,500,1000,2500];
settings.tolerance = tolerance;

%% INITIALIZE FIGURES
% Histograms for pixel intensities
intensities = struct();
histFig = [];  

% Main figure
if plotmode == "nBubbles"

    % Plot settings
    settings.figHeight = settings.figWidth; %Inches

    % Get the pulses
    filter = 'Reference';
    datasets = datasets(contains({datasets.name},filter));
    settings.order = [2,4,1,3];
    settings.cross_secs = settings.cross_secs(settings.order,:);

    % SELECT REGIONS OF INTEREST:
    % Near-field region:
    settings.nearFieldStartZ = 8;
    settings.nearFieldEndZ = 18;
    settings.nearFieldUpperX = 0;
    settings.nearFieldLowerX = 10;

    % Far-field region:
    settings.farFieldStartZ = 75;
    settings.farFieldEndZ = 85;
    settings.farFieldUpperX = 5;
    settings.farFieldLowerX = 15;

    % Mid-field region:
    settings.midFieldStartZ = 45;
    settings.midFieldEndZ = 55;
    settings.midFieldUpperX = 0;
    settings.midFieldLowerX = 10;

    % INITIALIZE TILEDLAYOUT
    bubFig = figure(50);
    bubFig.Units = "inches";
    bubFig.Position = [0 0 settings.figWidth settings.figHeight];
    bubTiles = tiledlayout(13,13,'TileSpacing','compact','Padding','tight');
    nexttile(bubTiles,1,[3 1]);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, 'Diffraction-limited', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', settings.fontSize, 'Rotation', 90,'BackgroundColor', 'none');
    nexttile(bubTiles,40, [3 1]);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, 'Super-resolved', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', settings.fontSize, 'Rotation', 90,'BackgroundColor', 'none');
    nexttile(bubTiles,79 , [3 1]);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, 'Contrast', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', settings.fontSize, 'Rotation', 90,'BackgroundColor', 'none');

elseif plotmode == "pulses"

    % Plot settings
    settings.figHeight = 4; % Height of the figure in inches

    % Get the pulses
    filt1 = contains({datasets.name},'pulseSingle_Reference_OneCycle_500bubbles');
    filt2 = contains({datasets.name},'pulseSingle_Short_MedF_500bubbles');
    filt3 = contains({datasets.name},'pulseSingle_Long_MedF_500bubbles');
    filt4 = contains({datasets.name},'pulseChirp_Long_Downsweep_500bubbles');
    filt5 = contains({datasets.name},'pulseChirp_Short_Downsweep_500bubbles');
    combinedFilter = boolean(filt1+filt2+filt3+filt4+filt5);
    datasets = datasets(combinedFilter);
    datasets = datasets([4,5,3,2,1]); % Put them in the right order
    settings.order = 1:5;

    % SELECT REGIONS OF INTEREST:
    % Near-field region:
    settings.nearFieldStartZ = 8;
    settings.nearFieldEndZ = 18;
    settings.nearFieldUpperX = 0;
    settings.nearFieldLowerX = 10;

    % Far-field region:
    settings.farFieldStartZ = 75;
    settings.farFieldEndZ = 85;
    settings.farFieldUpperX = 5;
    settings.farFieldLowerX = 15;

    % Mid-field region:
    settings.midFieldStartZ = 40;
    settings.midFieldEndZ = 50;
    settings.midFieldUpperX = -5;
    settings.midFieldLowerX = 5;

    % INITIALIZE TILEDLAYOUT
    pulFig = figure(50);
    pulFig.Units = "Inches";
    pulFig.Position = [0 0 settings.figWidth settings.figHeight];
    pulTiles = tiledlayout(6,16,'TileSpacing','Tight','Padding','tight');

    nexttile(pulTiles,1,[3 1]);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, '0% noise', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', settings.fontSize, 'Rotation', 90,'BackgroundColor', 'none');
    nexttile(pulTiles,49,[3 1]);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, '128% noise', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', settings.fontSize, 'Rotation', 90,'BackgroundColor', 'none');
end

%% LOOP TRHOUGH THE DATASETS

for dataset = 1 : length(datasets)

    dset = datasets(settings.order(dataset));

    substrings = strsplit(dset.name,'_');
    model = substrings(1) + "_" + substrings(2) + "_" + substrings(3);
    nBubbles = substrings(4);
    disp(model)
    disp(nBubbles)
    savedir = settings.parent_savedir + settings.delim + "model_" + model;

    %% LOAD THE RAW FILE
    path = settings.parent_path + settings.delim + dset.name;
    filename = 'RFDATA00001.mat';
    load(path+settings.delim+filename)

    % Compute FWHM
    FWHM = compute_FWHM(pulse.p_norm, pulse.tq(1, :));

    l_FWHM = FWHM*liquid.c*1e3;        % Convert to length scale im mm

    clear shell gas liquid

    %% SHOW THE DIFFRACION-LIMITED RECONSTRUCTION
    % Full diffraction-limited image

    filename_dl = strrep(filename, ".mat", "DAS_dl.mat");
    load(path+settings.delim+"Images"+settings.delim+filename_dl)

    % Reduce bubbles in vector file to the ones in the segment
    bubble_seg = bubble([bubble.z] > settings.midFieldStartZ*1e-3 & [bubble.z] < settings.midFieldEndZ*1e-3);
    bubble_seg = bubble_seg([bubble_seg.x] > settings.midFieldUpperX*1e-3 & [bubble_seg.x] < settings.midFieldLowerX*1e-3);

    super_resolved = false;
    showbubbles = true;

    % DEMODULATION
    % Demodulation of the signals
    if exist('img_log') ~= 1 % Log compress the image

        img_demod = zeros(size(img));
        for k = 1:size(img_demod,1)
            img_demod(k,:) = abs(hilbert(img(k,:)));
        end

        % LOG COMPRESSION
        maximg = max(max(img_demod));
        img_log = 20*log10(img_demod/maximg);
    else
        img = img_log;
        img_demod = 10.^(img_log / 20);
    end
    
    % DEFINE COLORBAR LIMITS
    clim_lower_dl = -30; %dB
    clim_upper_dl = 0; %dB

    % SHOW RECONSTRUCTION
    fig_dl = figure;
    ax_sr = show_reconstruction(z,x,img_log,super_resolved,bubble,showbubbles,clim_lower_dl,clim_upper_dl);

    % Highlight the regions of interest:
    hold(ax_sr, 'on')
    rectangle(ax_sr,'Position',[settings.midFieldStartZ settings.midFieldUpperX (settings.midFieldEndZ-settings.midFieldStartZ) (settings.midFieldLowerX-settings.midFieldUpperX)],'EdgeColor','r')
    ylim([-20 20]);
    hold(ax_sr, 'off')

    % Save the figure
    export_fig diffraction-limited_reconstruction.pdf
    sourceFile = "diffraction-limited_reconstruction.pdf";
    destinationFile = savedir + settings.delim + nBubbles + sourceFile;
    movefile(sourceFile, destinationFile)
    close(fig_dl);
    
    clear img_log img_demod img

    %% SHOW ZOOMED RECONSTRUCTION IN THE MID-FIELD
    % Show diffraction-limited midfield and put it in the nBubbleComparison
    % figure.

    if plotmode == "nBubbles"
        
        % CROSS-SECTION SETTINGS

        lat_cross_sec = settings.cross_secs(dataset,1);    % Lateral position of the cross-section in mm
        ax_cross_sec1 = settings.cross_secs(dataset,2);    % Axial start of the cross-section in mm
        ax_cross_sec2 = settings.cross_secs(dataset,3);    % Axial end of the cross-section in mm

        filename_dl_sec = strrep(filename, ".mat", "_section_bubbles_DAS_dl.mat");
        load(path+settings.delim+"Images"+settings.delim+filename_dl_sec)

        % Demodulation of the signals
        if exist('img_log') ~= 1

            img_demod = zeros(size(img));
            for k = 1:size(img_demod,1)
                img_demod(k,:) = abs(hilbert(img(k,:)));
            end

            % LOG COMPRESSION
            maximg = max(max(img_demod));
            img_log = 20*log10(img_demod/maximg);
        else
            img = img_log;
            img_demod = 10.^(img_log / 20);
        end

        % Tiledlayout plot
        figure(bubFig)
        ax((dataset-1)*2+1) = show_reconstruction_tiledlayout(z_rec,x_rec,img_log,super_resolved,bubble_seg,showbubbles,clim_lower_dl,clim_upper_dl,(dataset-1)*3+2);
        title(string(settings.nBubblesList(dataset)))

        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize', settings.fontSize,'FontName', 'Times New Roman')
        end
        plot([ax_cross_sec1,ax_cross_sec2],[lat_cross_sec,lat_cross_sec],"--", Color="#EDB120")
        rectangle('Position',[ax_cross_sec1 (lat_cross_sec-0.5) (ax_cross_sec2-ax_cross_sec1) 1], 'EdgeColor','w')
        xlim([settings.midFieldStartZ settings.midFieldEndZ])
        ylim([settings.midFieldUpperX settings.midFieldLowerX])
        set(gca,'FontSize',settings.fontSize,'FontName', 'Times New Roman')

        % SHOW CROSS-SECTION INTENSITIES

        % Convert to pixels
        [~,I] = min(abs((lat_cross_sec*1e-3)-x_rec));
        [~,J1] = min(abs((ax_cross_sec1*1e-3)-z_rec));
        [~,J2] = min(abs((ax_cross_sec2*1e-3)-z_rec));

        y = img_demod(I,J1:J2);
        figure(7)
        plot(z(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),Color="#EDB120")
        hold on

        figure('Visible','off');
        plot((y-min(y))/(max(y)-min(y)),Color="#EDB120")

        % Using tiledlayout
        figure(bubFig)
        loc_tile = nexttile(bubTiles,(dataset-1)*3+80,[1 3]);
        loc_tile = disp_bub_location_small(loc_tile, bubble_seg, ax_cross_sec1, ax_cross_sec2, lat_cross_sec);

        int_tile = nexttile(bubTiles,(dataset-1)*3+93,[2 3]);
        plot(z_rec(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),Color="#EDB120")
        hold on

    end

    %% SHOW THE SUPER-RESOLVED RECONSTRUCTION
    % Full super-resolved image

    if exist("tolerance") == 1
        filename_sr = strrep(filename, ".mat", "DAS_sr_tol"+ string(settings.tolerance)+".mat");
    else
        filename_sr = strrep(filename, ".mat", "DAS_sr.mat");
    end

    load(path+settings.delim+"Images"+ settings.delim + filename_sr)

    img_demod = abs(img);
    super_resolved = true;

    % COMPUTE THE COLORBAR LIMITS
    img_demod2 = img_demod(abs(x_rec)<(max(domain.x_el)),:);    % Remove pixels outside domain
    img_demod2(img_demod2<0.001)=nan;                       % Set the zeros to nan

    % Define colorbar limits
    pd = fitdist(img_demod2(:),'normal');
    clim_lower_sr = pd.mu + 1*pd.sigma;
    clim_upper_sr = pd.mu + 8*pd.sigma;
    intensities(dataset).clim_lower_sr = clim_lower_sr;
    intensities(dataset).clim_upper_sr = clim_upper_sr;

    % SHOW RECONSTRUCTION
    fig_sr = figure(4);
    show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr);

    % Highlight the regions of interest:
    hold on
    rectangle('Position',[settings.midFieldStartZ settings.midFieldUpperX (settings.midFieldEndZ-settings.midFieldStartZ) (settings.midFieldLowerX-settings.midFieldUpperX)],'EdgeColor','r')
    ylim([-20 20])

    export_fig super-resolved_reconstruction.pdf
    sourceFile = "super-resolved_reconstruction.pdf";
    destinationFile = savedir + settings.delim + nBubbles + sourceFile;
    movefile(sourceFile, destinationFile)
    close(fig_sr);

    %% SHOW ZOOMED RECONSTRUCTION IN THE MID-FIELD
    % The mid-field reconstructions are used in nBubblesComparison and
    % pulseComparison.

    if plotmode == "nBubbles"
        % nBubbles figure
        
        if exist("tolerance") == 1
            filename_sr = strrep(filename, ".mat", "_section_bubbles_DAS_sr_tol"+ string(settings.tolerance)+".mat");
        else
            filename_sr = strrep(filename, ".mat", "_section_bubbles_DAS_sr.mat");
        end
        load(path+settings.delim+"Images"+settings.delim+filename_sr)
        
        img_demod = abs(img);
        super_resolved = true;

        % COMPUTE THE COLORBAR LIMITS
        img_demod2 = img_demod(abs(x_rec)<(max(domain.x_el)),:);    % Remove pixels outside domain
        img_demod2(img_demod2<0.001)=nan;                       % Set the zeros to nan

        pd = fitdist(img_demod2(:),'normal');
        disp('Intensity of noise free image')
        disp(pd)
        clim_lower_sr = pd.mu + 1*pd.sigma;
        clim_upper_sr = pd.mu + 8*pd.sigma;
        
        % Store intensities in struct
        intensities(dataset).values = img_demod2;
        intensities(dataset).num_bub = nBubbles;

        % Using tiledlayout
        figure(bubFig)
        show_reconstruction_tiledlayout(z_rec,x_rec,img_demod,super_resolved,bubble_seg,showbubbles,clim_lower_sr,clim_upper_sr,(dataset-1)*3+41);

        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',settings.fontSize,'FontName', 'Times New Roman')
        end
        plot([ax_cross_sec1,ax_cross_sec2],[lat_cross_sec,lat_cross_sec],"--", Color="b")

        rectangle('Position',[ax_cross_sec1 (lat_cross_sec-0.5) (ax_cross_sec2-ax_cross_sec1) 1], 'EdgeColor','k')
        xlim([settings.midFieldStartZ settings.midFieldEndZ])
        ylim([settings.midFieldUpperX settings.midFieldLowerX])
        tex = text((settings.midFieldEndZ-settings.midFieldStartZ)*0.8+settings.midFieldStartZ, (settings.midFieldLowerX-settings.midFieldUpperX)*0.9+settings.midFieldUpperX,"tol = "+string(settings.tolerance),'Color','k', 'FontSize', settings.fontSize,'FontName','Times New Roman');
        set(gca,'FontSize',settings.fontSize,'FontName', 'Times New Roman')

    elseif plotmode == "pulses"
        % pulseComparison figure

        % Without noise
        if exist("tolerance") == 1
            filename_sr = strrep(filename, ".mat", "_section_pulses_DAS_sr_tol"+ string(settings.tolerance)+".mat");
        else
            filename_sr = strrep(filename, ".mat", "_section_pulses_DAS_sr.mat");
        end
        load(path+settings.delim+"Images"+settings.delim+filename_sr)
        disp("Loaded: "+path+settings.delim+"Images"+settings.delim+filename_sr)

        img_demod = abs(img);

        % COMPUTE THE COLORBAR LIMITS
        img_demod2 = img_demod(abs(x_rec)<(max(domain.x_el)),:);    % Remove pixels outside domain
        img_demod2(img_demod2<0.001)=nan;                       % Set the zeros to nan

        pd = fitdist(img_demod2(:),'normal');
        clim_lower_sr = pd.mu + 1*pd.sigma;
        clim_upper_sr = pd.mu + 8*pd.sigma;

        % Using tiledlayout
        figure(pulFig)
        show_reconstruction_tiledlayout(z_rec,x_rec,img_demod,super_resolved,bubble_seg,showbubbles,clim_lower_sr,clim_upper_sr,(dataset-1)*3+2);

        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',settings.fontSize,'FontName', 'Times New Roman')
        end
        title(model)
        xlim([settings.midFieldStartZ settings.midFieldEndZ])
        ylim([settings.midFieldUpperX settings.midFieldLowerX])
        text((settings.midFieldEndZ-settings.midFieldStartZ)*0.75+settings.midFieldStartZ, (settings.midFieldLowerX-settings.midFieldUpperX)*0.9+settings.midFieldUpperX,"tol = "+string(settings.tolerance),'Color','k', 'FontSize', settings.fontSize,'FontName','Times New Roman');
        set(gca,'FontSize',settings.fontSize,'FontName', 'Times New Roman')
   
        % Image with noise
        if exist("tolerance") == 1
            filename_sr = strrep(filename, ".mat", "_section_pulses_DAS_sr_tol"+ string(settings.tolerance)+additional_specification+".mat");
        else
            filename_sr = strrep(filename, ".mat", "_section_pulses_DAS_sr" + additional_specification + ".mat");
        end

        load(path + settings.delim + "Images" + settings.delim + filename_sr)
        disp("Loaded: "+path+settings.delim+"Images"+settings.delim+filename_sr)
        img_demod = abs(img);
        super_resolved = true;

        % Compute intensities
        % Set the zeros to nan
        img_demod2 = img_demod;
        img_demod2(img_demod2<0.001)=nan;

        % Store intensities in struct
        intensities(dataset).values = img_demod2;
        intensities(dataset).num_bub = nBubbles;

        % Define colorbar limits
        pd = fitdist(img_demod2(:),'normal');
        clim_lower_sr = pd.mu + 1*pd.sigma;
        clim_upper_sr = pd.mu + 8*pd.sigma;

        % Using tiledlayout
        figure(pulFig)
        show_reconstruction_tiledlayout(z_rec,x_rec,img_demod,super_resolved,bubble_seg,showbubbles,clim_lower_sr,clim_upper_sr,(dataset-1)*3+50);
        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',settings.fontSize,'FontName', 'Times New Roman')
        end
        xlim([settings.midFieldStartZ settings.midFieldEndZ])
        ylim([settings.midFieldUpperX settings.midFieldLowerX])
        tex = text((settings.midFieldEndZ-settings.midFieldStartZ)*0.75+settings.midFieldStartZ, (settings.midFieldLowerX-settings.midFieldUpperX)*0.9+settings.midFieldUpperX,"tol = "+string(settings.tolerance),'Color','k', 'FontSize', settings.fontSize,'FontName','Times New Roman');
        set(gca,'FontSize',settings.fontSize,'FontName', 'Times New Roman')

    end

    % SHOW CROSS-SECTION INTENSITIES
    if plotmode == "nBubbles"
        y = img_demod(I,J1:J2);
        
        % Compute the FWHM
        FWHM = compute_FWHM(y,z_rec(J1:J2)*1000);
        disp("FWHM for the super-resolved image of " + settings.nBubblesList(dataset) + " bubbles is " + FWHM + "mm")

        figure(7)
        hold on
        axis square
        plot(z_rec(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),'b')
        xlim([ax_cross_sec1 ax_cross_sec2])
        ylim([0,1])
        yticks(0:0.5:1)
        xlabel('z [mm]')
        ylabel('Intensity (normalized)')

        legend('diffraction-limited', 'super-resolved')
        set(gca, 'Units', 'inches');
        set(gca, 'Position', [1 1 2.358 2.358], 'Units','inches')
        set(gca, 'FontSize', 8,'FontName', 'Times New Roman');

        exportgraphics(gcf, savedir + settings.delim + nBubbles + "_crosssection.pdf", 'ContentType', 'vector', 'Resolution', settings.dpi)
        close;

        % Using tiledlayout
        figure(bubFig)
        nexttile(bubTiles, (dataset-1)*3+93)
        hold on
        plot(z_rec(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),'b')
        xlim([ax_cross_sec1 ax_cross_sec2])
        ylim([0,1])
        yticks(0:0.5:1)
        set(gca,'FontSize',settings.fontSize,'FontName', 'Times New Roman')
        % axis square
    end

end


%% INTENSITY HISTOGRAM FIGURE
% Create a histogram with the intensity values of the plot. This is used
% to provide information on the colorbar limits.

if plotmode == "nBubbles"
    % Show histogram of the intensity values in the super-resolved image
    histFig = figure('PaperPositionMode', 'manual','units','inch','position',[0,0,3.5,3]);
    h = tiledlayout(histFig,2,2);

    for d = 1:length(datasets)     % Reopen figure
        ax = nexttile;
        h = histogram(intensities(d).values);
        r = rectangle('Position', [intensities(d).clim_lower_sr  min(ax.YLim)  (intensities(d).clim_upper_sr-intensities(d).clim_lower_sr)  max(ax.YLim)],'FaceColor',[0, 0, 1, 0.2]);
        title(string(settings.nBubblesList(d)) + " bubbles")
        grid on
        xlabel('intensity (a.u.)')
        ylabel('counts')
        xlim([0,1])
        set(gca,'FontSize',settings.fontSize,'FontName', 'Times New Roman')
    end
    histFig.Color = 'w';
    exportgraphics(gcf,settings.parent_savedir + settings.delim +"intensity_histogram.pdf", 'ContentType', 'vector', 'Resolution', settings.dpi)
end

%% FIGURE FORMATTING
% Format the main figures

if plotmode == "nBubbles"

    figure(bubFig);
    title(bubTiles,'Number of bubbles','FontSize', settings.fontSize, 'FontName', 'Times New Roman');
    xlabel(bubTiles,'Lateral distance [mm]','FontSize', settings.fontSize);
    bubFig.Color = 'w';
    export_fig nBubbleComparison_frame -svg -painters
    sourceFile = "nBubbleComparison_frame.svg";
    destinationFile = settings.parent_savedir + settings.delim + sourceFile;
    movefile(sourceFile, destinationFile)

    % Export the images
    bubFig.Position = get(0,'screensize');
    export_fig nBubbleComparison_images.pdf -native
    sourceFile = "nBubbleComparison_images.pdf";
    destinationFile = settings.parent_savedir + settings.delim + sourceFile;
    movefile(sourceFile, destinationFile)
elseif plotmode == "pulses"

    % Export the formatted axes
    figure(pulFig);
    title(pulTiles,'Pulse type','FontSize', settings.fontSize, 'FontName', 'Times New Roman');
    xlabel(pulTiles,'Lateral distance [mm]','FontSize', settings.fontSize);
    export_fig PulseComparison_frame -svg -painters
    sourceFile = "PulseComparison_frame.svg";
    destinationFile = settings.parent_savedir + settings.delim + sourceFile;
    movefile(sourceFile, destinationFile)

    % Export the images
    pulFig.Position = get(0,'screensize');
    export_fig PulseComparison_images.pdf -native
    sourceFile = "PulseComparison_images.pdf";
    destinationFile = settings.parent_savedir + settings.delim + sourceFile;
    movefile(sourceFile, destinationFile)
end

%% FUNCTIONS
function ax = disp_bub_location_small(ax, bubble, ax_cross_sec1, ax_cross_sec2, lat_cross_sec)
hold(ax, 'on')

% Plot the bubbles in the subsection
for k = 1:length(bubble)
    plot(ax, bubble(k).z*1e3, bubble(k).x*1e3, 'ro', 'MarkerSize', 3)
end
plot([ax_cross_sec1,ax_cross_sec2],[lat_cross_sec,lat_cross_sec],"--", Color="#EDB120")

% Format the axis
ax.XLim = [ax_cross_sec1 ax_cross_sec2];
ax.YLim = [(lat_cross_sec-0.5) (lat_cross_sec+0.5)];
box on;
set(ax,'xtick',[]);
set(ax,'ytick',[]);
set(ax, 'YDir','reverse') % Revert the Y-axis
end

function FWHM = compute_FWHM(signal,ax_values)
    % Compute the Full Width Half Maximum of the signal on ax_values. This
    % function only looks at the halfmax around the peak in signal.

    % Upsample the signal and the axis values
    upsampling_factor = 10;
    ax_values_new = ax_values(1):(unique(diff(ax_values)))/upsampling_factor:ax_values(end);
    signal_new = interp1(ax_values, signal, ax_values_new);
    
    % Locate the max
    [~,I] = max(abs(hilbert(signal_new)));

    % Compute halfMax
    halfMax = (min(abs(hilbert(signal_new))) + max(abs(hilbert(signal_new)))) / 2;
    
    % Find halfmax index lower than max
    index1 = I;

    while signal_new(index1) > halfMax
        index1 = index1 - 1;
    end
    
    index1 = index1 + 1;

    % Find halfmax index larger than max
    index2 = I;

    while signal_new(index2) > halfMax
        index2 = index2 + 1;
    end
    
    index2 = index2 - 1;
    
    FWHM = ax_values_new(index2) - ax_values_new(index1);
end