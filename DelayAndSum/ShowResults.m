% Script to create the delay-and-sum figures in the manuscript.

close all

delim = "\\";

% Load the metadata from the original MATLAB files:
parent_path = "D:\SRML-1D-pulse-types\Results\RF signals\mat_files2D";
parent_savedir = "D:\SRML-1D-pulse-types\Results\Figures";
datasets = dir(parent_path);
datasets = datasets(3:end);

% Add path to export_fig module
addpath('C:\Users\rienk\OneDrive\Bureaublad\export_fig-master')

%% PLOT SETTINGS
% Plot type
plotmode = "pulses"; % Enter "nBubbles", "pulses" or "all"
additional_specification = "_noise128";

% Figure settings
dpi = 600;
figWidth = 7.16; %Inches
fontSize = 6;

% Specify cross-sections
cross_secs = [1.64553,49,52;
    1.64375,50,53;
    1.93755,47,50;
    1.36694,47,50;]; % For densities of resp. 1000,100,2500,500. This is chosen such the cross section is at exactly one bubble.

nBubblesList = [100,500,1000,2500];
histFig = [];
intensities = struct();

if plotmode == "nBubbles"
    % Plot settings
    figHeight = 6; %Inches

    % Get the pulses
    filter = 'Reference';
    datasets = datasets(contains({datasets.name},filter));
    order = [2,4,1,3];
    cross_secs = cross_secs(order,:);

    % SELECT REGIONS OF INTEREST:
    % Near-field region:
    x1_1 = 8;
    x1_2 = 18;
    y1_1 = 0;
    y1_2 = 10;

    % Far-field region:
    x2_1 = 75;
    x2_2 = 85;
    y2_1 = 5;
    y2_2 = 15;

    % Mid-field region:
    x3_1 = 45;
    x3_2 = 55;
    y3_1 = 0;
    y3_2 = 10;

    % INITIALIZE TILEDLAYOUT
    bubFig = figure(50);
    bubFig.Units = "inches";
    bubFig.Position = [0 0 figWidth figHeight];
    bubTiles = tiledlayout(3,5,'TileSpacing','Tight','Padding','tight');
    nexttile(bubTiles,1);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, 'Diffraction-limited', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', fontSize, 'Rotation', 90,'BackgroundColor', 'none'); 
    nexttile(bubTiles,6);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, 'Super-resolved', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', fontSize, 'Rotation', 90,'BackgroundColor', 'none');
    nexttile(bubTiles,11);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, 'Contrast', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', fontSize, 'Rotation', 90,'BackgroundColor', 'none');

elseif plotmode == "pulses"
    % Plot settings
    figHeight = 4; %Inches

    % Get the pulses
    filt1 = contains({datasets.name},'pulseSingle_Reference_OneCycle_500bubbles');
    filt2 = contains({datasets.name},'pulseSingle_Short_MedF_500bubbles');
    filt3 = contains({datasets.name},'pulseSingle_Long_MedF_500bubbles');
    filt4 = contains({datasets.name},'pulseChirp_Long_Downsweep_500bubbles');
    filt5 = contains({datasets.name},'pulseChirp_Short_Downsweep_500bubbles');
    combinedFilter = boolean(filt1+filt2+filt3+filt4+filt5);
    datasets = datasets(combinedFilter);
    datasets = datasets([4,5,3,2,1]);
    order = 1:5;

    % SELECT REGIONS OF INTEREST:
    % Near-field region:
    x1_1 = 8;
    x1_2 = 18;
    y1_1 = 0;
    y1_2 = 10;

    % Far-field region:
    x2_1 = 75;
    x2_2 = 85;
    y2_1 = 5;
    y2_2 = 15;

    % Mid-field region:
    x3_1 = 40;
    x3_2 = 50;
    y3_1 = -5;
    y3_2 = 5;

    % INITIALIZE TILEDLAYOUT
    pulFig = figure(50);
    pulFig.Units = "Inches";
    pulFig.Position = [0 0 figWidth figHeight];
    pulTiles = tiledlayout(2,6,'TileSpacing','Tight','Padding','tight');

    nexttile(pulTiles,1);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, '0% noise', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', fontSize, 'Rotation', 90,'BackgroundColor', 'none');
    nexttile(pulTiles,7);
    ax = gca; % Get the current axis
    ax.XColor = 'none'; % Hide x-axis
    ax.YColor = 'none'; % Hide y-axis
    ax.Color = 'none';
    axis square
    text(0.5, 0.5, '128% noise', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', fontSize, 'Rotation', 90,'BackgroundColor', 'none');
end


for dataset = 1 : length(datasets)

    dset = datasets(order(dataset));

    substrings = strsplit(dset.name,'_');
    model = substrings(1) + "_" + substrings(2) + "_" + substrings(3);
    nBubbles = substrings(4);
    disp(model)
    disp(nBubbles)
    savedir = parent_savedir + delim + "model_" + model;

    %% LOAD THE RAW FILE
    path = parent_path + delim + dset.name;
    filename = 'RFDATA00001.mat';
    load(path+delim+filename)

    % Compute FWHM
    t = pulse.tq(1,:);
    halfMax = (min(abs(hilbert(pulse.p_norm))) + max(abs(hilbert(pulse.p_norm)))) / 2;
    index1 = find(pulse.p_norm >= halfMax,1,"first");
    index2 = find(pulse.p_norm >= halfMax,1,"last");
    FWHM = t(index2) - t(index1);

    l_FWHM = FWHM*liquid.c*1e3;        % Convert to length scale im mm

    %% SHOW THE DIFFRACION-LIMITED RECONSTRUCTION:
    filename_sr = dset.name + "DAS";
    load(path+delim+filename_sr)

    super_resolved = false;
    showbubbles = true;

    % DEMODULATION
    % Demodulation of the signals
    img_demod = zeros(size(img));
    for k = 1:size(img_demod,1)
        img_demod(k,:) = abs(hilbert(img(k,:)));
    end

    % CROSS-SECTION SETTINGS
    if plotmode == "nBubbles"
        lat_cross_sec = cross_secs(dataset,1);     % Lateral position of the cross-section in mm
        ax_cross_sec1 = cross_secs(dataset,2);    % Axial start of the cross-section in mm
        ax_cross_sec2 = cross_secs(dataset,3);    % Axial end of the cross-section in mm

        % Convert to pixels
        I  = int32(1e-3*lat_cross_sec*(size(img_demod,1)/2)/(domain.width*1.5/2))+size(img_demod,1)/2;
        J1 = int32(1e-3*ax_cross_sec1*(size(img_demod,2)/(domain.depth)));
        J2 = int32(1e-3*ax_cross_sec2*(size(img_demod,2)/(domain.depth)));
    end

    % LOG COMPRESSION
    maximg = max(max(img_demod));
    img_log = 20*log10(img_demod/maximg);
    disp(strcat('Maximum value: ', num2str(max(max(img_log))), 'dB'))
    disp(strcat('Minimum value: ', num2str(min(min(img_log))), 'dB'))

    % DEFINE COLORBAR LIMITS
    clim_lower_dl = -30; %dB
    clim_upper_dl = 0; %dB

    % SHOW RECONSTRUCTION
    fig_dl = figure(1);
    show_reconstruction(z,x,img_log,super_resolved,bubble,showbubbles,clim_lower_dl,clim_upper_dl);

    % Highlight the regions of interest:
    hold on
    rectangle('Position',[x3_1 y3_1 (x3_2-x3_1) (y3_2-y3_1)],'EdgeColor','r')
    ylim([-20 20]);

    % Save the figure
    export_fig diffraction-limited_reconstruction.pdf
    sourceFile = "diffraction-limited_reconstruction.pdf";
    destinationFile = savedir + delim + nBubbles + sourceFile;
    movefile(sourceFile, destinationFile)
    close(fig_dl);

    % SHOW ZOOMED RECONSTRUCTION IN THE MID-FIELD

    if plotmode == "nBubbles"
        % Tiledlayout plot
        figure(bubFig)
        show_reconstruction_tiledlayout(z,x,img_log,super_resolved,bubble,showbubbles,clim_lower_dl,clim_upper_dl,dataset+1);
        title(string(nBubblesList(dataset)))

        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',fontSize)
        end
        plot([ax_cross_sec1,ax_cross_sec2],[lat_cross_sec,lat_cross_sec],"--", Color="#EDB120")
        xlim([x3_1 x3_2])
        ylim([y3_1 y3_2])
        set(gca,'FontSize',fontSize)

        % SHOW CROSS-SECTION INTENSITIES
        y = img_demod(I,J1:J2);
        figure(7)
        plot(z(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),Color="#EDB120")
        hold on
        
        figure(200+dataset);
        plot((y-min(y))/(max(y)-min(y)),Color="#EDB120")

        % Using tiledlayout
        figure(bubFig)
        int_tile = nexttile(bubTiles,dataset+11);
        plot(z(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),Color="#EDB120")
        hold on

    end

    %% SHOW THE SUPER-RESOLVED RECONSTRUCTION
    load(path + delim + dset.name + "DAS_sr")
    img_demod = abs(img);
    super_resolved = true;

    disp(strcat('Maximum value: ', num2str(max(max(img_demod)))))
    disp(strcat('Minimum value: ', num2str(min(min(img_demod)))))

    % COMPUTE THE COLORBAR LIMITS
    % Set the zeros to nan
    img_demod2 = img_demod;
    img_demod2(img_demod2<0.1)=nan;

    % Store intensities in struct
    intensities(dataset).values = img_demod2;
    intensities(dataset).num_bub = nBubbles;

    % Define colorbar limits
    pd = fitdist(img_demod2(:),'normal');
    disp('Intensity of noise free image')
    disp(pd)
    clim_lower_sr = pd.mu + 1*pd.sigma;
    clim_upper_sr = pd.mu + 8*pd.sigma;

    % SHOW RECONSTRUCTION
    fig_sr = figure(4);
    show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr);

    % Highlight the regions of interest:
    hold on
    rectangle('Position',[x3_1 y3_1 (x3_2-x3_1) (y3_2-y3_1)],'EdgeColor','r')
    ylim([-20 20])

    export_fig super-resolved_reconstruction.pdf
    sourceFile = "super-resolved_reconstruction.pdf";
    destinationFile = savedir + delim + nBubbles + sourceFile;
    movefile(sourceFile, destinationFile)
    close(fig_sr);

    % SHOW ZOOMED RECONSTRUCTION IN THE MID-FIELD
    if plotmode == "nBubbles"

        % Using tiledlayout
        figure(bubFig)
        show_reconstruction_tiledlayout(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr,dataset+6);
        
        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',fontSize)
        end
        plot([ax_cross_sec1,ax_cross_sec2],[lat_cross_sec,lat_cross_sec],"--", Color="b")
        xlim([x3_1 x3_2])
        ylim([y3_1 y3_2])
        set(gca,'FontSize',fontSize)
        
    elseif plotmode == "pulses"

        % Using tiledlayout
        figure(pulFig)
        show_reconstruction_tiledlayout(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr,dataset+1)
        
        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',fontSize)
        end
        title(model)
        xlim([x3_1 x3_2])
        ylim([y3_1 y3_2])
        set(gca,'FontSize',fontSize)
    end
    
    %% SHOW THE SUPER-RESOLVED RECONSTRUCTION
    if plotmode == "pulses"

        load(path + delim + dset.name + "DAS_sr" + additional_specification)
        disp('loading pulse with noise')
        img_demod = abs(img);
        super_resolved = true;
        
        %% Compute intensities
        % Set the zeros to nan
        img_demod2 = img_demod;
        img_demod2(img_demod2<0.1)=nan;

        % Store intensities in struct
        intensities(dataset).values = img_demod2;
        intensities(dataset).num_bub = nBubbles;

        % Define colorbar limits
        pd = fitdist(img_demod2(:),'normal');
        disp('Image intensity with 128% noise')
        disp(pd)

        % SHOW RECONSTRUCTION
        fig_sr_noise = figure(11);
        show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr)

        % Highlight the regions of interest:
        hold on
        rectangle('Position',[x3_1 y3_1 (x3_2-x3_1) (y3_2-y3_1)],'EdgeColor','r')
        ylim([-20 20])
        
        % Plot the bar indicating the length of the pulse
        if model == "pulseSingle_Reference_OneCycle"
            x_start = x3_2-l_FWHM-0.5;
            x_end = x3_2-0.5;
            y = y3_2-0.5;
            plot([x_start x_end],[y y], 'w', 'LineWidth', 3)
        end

        % Export the figure
        export_fig super-resolved_reconstruction_noise128.pdf
        sourceFile = "super-resolved_reconstruction_noise128.pdf";
        destinationFile = savedir + delim + nBubbles + sourceFile;
        movefile(sourceFile, destinationFile)
        close(fig_sr_noise);

        % % SHOW ZOOMED RECONSTRUCTION IN THE MID-FIELD

        img_demod2 = img_demod;
        img_demod2(img_demod2<0.1)=nan;

        figure(100+dataset)
        h = histogram(img_demod2);
        title([model, nBubbles])
        xlabel('intensity (a.u.)')
        ylabel('counts')
        exportgraphics(gcf,savedir + delim + nBubbles + "_intensity_histogram_noise128.pdf", 'ContentType', 'vector', 'Resolution', dpi)
        
        % Using tiledlayout
        figure(pulFig)
        show_reconstruction_tiledlayout(z,x,img_demod,super_resolved,bubble,showbubbles,clim_lower_sr,clim_upper_sr,dataset+7)
        if dataset == 1
            ylabel('Lateral distance [mm]', 'FontSize',fontSize)
        end
        xlim([x3_1 x3_2])
        ylim([y3_1 y3_2])
        set(gca,'FontSize',fontSize)

    end

    % SHOW CROSS-SECTION INTENSITIES
    if plotmode == "nBubbles"
        y = img_demod(I,J1:J2);
        figure(7)
        hold on
        axis square
        plot(z(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),'b')
        xlim([ax_cross_sec1 ax_cross_sec2])
        ylim([0,1])
        yticks(0:0.5:1)
        xlabel('z [mm]')
        ylabel('Intensity (normalized)')

        legend('diffraction-limited', 'super-resolved')
        set(gca, 'Units', 'inches');
        set(gca, 'Position', [1 1 2.358 2.358], 'Units','inches')
        set(gca, 'FontSize', 8);

        exportgraphics(gcf, savedir + delim + nBubbles + "_crosssection.pdf", 'ContentType', 'vector', 'Resolution', dpi)
        close;

        % Using tiledlayout
        figure(bubFig)
        nexttile(bubTiles, dataset+11)
        hold on
        plot(z(J1:J2)*1000,(y-min(y))/(max(y)-min(y)),'b')
        if dataset == 1
            ylabel('Intensity (normalized)', 'FontSize',fontSize)
        end
        xlim([ax_cross_sec1 ax_cross_sec2])
        ylim([0,1])
        yticks(0:0.5:1)
        set(gca,'FontSize',fontSize)
        axis square
    end

end

if plotmode == "nBubbles"
    % Show histogram of the intensity values in the super-resolved image
    histFig = figure('units','inch','position',[0,0,3.5,4]);
    h = tiledlayout(histFig,2,2);

    for d = 1:length(datasets)     % Reopen figure
        nexttile
        histogram(intensities(d).values);
        title(intensities(d).num_bub)
        grid on
        xlabel('intensity (a.u.)')
        ylabel('counts')
        xlim([-5,70])
        set(gca,'FontSize',fontSize)
    end

    exportgraphics(gcf,savedir + delim +"intensity_histogram.pdf", 'ContentType', 'vector', 'Resolution', dpi)

    figure(bubFig);
    title(bubTiles,'Number of bubbles','FontSize', fontSize);
    xlabel(bubTiles,'Lateral distance [mm]','FontSize', fontSize);
    export_fig nBubbleComparison.pdf
    sourceFile = "nBubbleComparison.pdf";
    destinationFile = parent_savedir + delim + sourceFile;
    movefile(sourceFile, destinationFile)
elseif plotmode == "pulses"
    figure(pulFig);
    title(pulTiles,'Pulse type','FontSize', fontSize);
    xlabel(pulTiles,'Lateral distance [mm]','FontSize', fontSize);
    export_fig PulseComparison.pdf
    sourceFile = "PulseComparison.pdf";
    destinationFile = parent_savedir + delim + sourceFile;
    movefile(sourceFile, destinationFile)
end

%% FUNCTIONS

function show_reconstruction(z,x,img,super_resolved,bubble,showbubbles,c1,c2)
% Show the reconstructed image

if super_resolved == false
    fig_title = 'Normal resolution';        % Title for the figure
    colob_title = 'image intensity (dB)';   % Title for the colorbar
    cmap = 'gray';                          % Colormap

else
    cmap = 'gray';
    cmap = colormap(cmap);
    cmap = colormap(flipud(cmap));          % Invert colormap

    fig_title = 'Super-resolved';           % Title for the figure
    colob_title = ['image intensity,'...    % Title for the colorbar
        newline 'linear scale (a.u.)'];
end


imagesc(z.*1e3,x.*1e3,img);

if showbubbles
    % Show the bubbles
    hold on
    for k = 1:length(bubble)
        plot(bubble(k).z*1e3, bubble(k).x*1e3, 'ro')
    end

    %legend('Bubbles')
end

clim([c1 c2]);
colob = colorbar;
ylabel('lateral distance [mm]','interpreter', 'latex','fontsize',16)
xlabel('axial distance [mm]','interpreter', 'latex','fontsize',16)
ylabel(colob,colob_title,'interpreter', 'latex','fontsize',16);
title(fig_title)
axis equal
drawnow
colormap(cmap);
end

function show_reconstruction_tiledlayout(z,x,img,super_resolved,bubble,showbubbles,c1,c2,n)
% Show the reconstructed image in a tiled layout.
if super_resolved == false
    fig_title = 'Normal resolution';        % Title for the figure
    colob_title = 'image intensity (dB)';   % Title for the colorbar
    cmap = 'gray';                          % Colormap
else
    cmap = 'gray';
    cmap = flipud(gray);                    % Invert colormap
    fig_title = 'Super-resolved';           % Title for the figure
    colob_title = ['image intensity,'...    % Title for the colorbar
        newline 'linear scale (a.u.)'];
end

% Create the tile and plot
ax = nexttile(n);
imagesc(ax, z.*1e3, x.*1e3, img);

if showbubbles
    % Show the bubbles
    hold(ax, 'on')
    for k = 1:length(bubble)
        plot(ax, bubble(k).z*1e3, bubble(k).x*1e3, 'ro', 'MarkerSize', 3)
    end
end

% Configure axes properties
axis(ax, 'equal')
clim(ax, [c1 c2]);
colormap(ax, cmap);

% Optional: Uncomment if needed
% colob = colorbar(ax);
% ylabel(ax, 'lateral distance [mm]', 'interpreter', 'latex', 'fontsize', 16)
% xlabel(ax, 'axial distance [mm]', 'interpreter', 'latex', 'fontsize', 16)
% ylabel(colob, colob_title, 'interpreter', 'latex', 'fontsize', 8);
% title(ax, fig_title)

drawnow

end