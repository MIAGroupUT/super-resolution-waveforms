% Plots the pulses used in the simulations
%
% Rienk Zorgdrager, University of Twente, 2024

%% CLEAR AND CLOSE ALL
clear
clc

%% ADD PATHS
% Add the functions folder to path
addpath './functions'
addpath './microbubble-simulator'
addpath './microbubble-simulator/functions'

% Get the transducer transfer functions of the P4-1 transducer
Tfit = load('TransmitTransferFunctionFit.mat');

%% FIGURE SETTINGS
figHeight = 4;      % Height in inches
figWidth = 7.16;    % Width in inches
LineWidth = 0.5;    % Linewidth
FontName = 'times';
FontSize = 10;      % Font size in pt
dpi = 600;          % Dots per inch

dispFig = false;

%% PULSE PROPERTIES
% Get the pulses
pulseProperties = get_pulse_properties();

% Simulation and recording sampling rate:
Fs = Tfit.Fs;              	% Simulation sampling rate (Hz)
pulseProperties.Fs = Fs;

%% GET TRANSMISSION OUTPUTS
% Calculate the driving time, input voltage and normalized pressure outcome
[pulses, pulseSequence] = getPulseInOutput(pulseProperties, Tfit, dispFig);

%% MAKE THE FIGURE
nPulses = length(pulseSequence);
nRows = 2;
nCols = ceil(nPulses/nRows);
t_max = max(pulses.pulseTrain.Short.Delay.t);

% Locate the pulses on the desired location
idx = [2,3,4,8,10,1,5,6,7,9,11,12];
pulseSequence = pulseSequence(idx);

fig = figure();
fig.Units = "inches";
fig.Position(3:4) = [7.16,4];
t = tiledlayout(nRows,nCols,"TileSpacing","none");


% Plot the results
for nPulse=1:nPulses
    % Load the pulse
    acronym  = pulseSequence(nPulse).acronym;
    pulseCat = pulseSequence(nPulse).pulseCategory;
    lenCat   = pulseSequence(nPulse).lengthCategory;
    varCat   = pulseSequence(nPulse).variableCategory;

    pulse = pulses.(pulseCat).(lenCat).(varCat);
    t = pulse.t + (t_max/2) - max(pulse.t)/2 ;
    p_norm = pulse.p_norm;

    % Compute pulse duration (full width at half maximum)
    % Calculate FWHM
    H = abs(hilbert(p_norm));
    index1 = find(H >= 0.5,1,"first");
    index2 = find(H >= 0.5,1,"last");
    pulse.FWHM = pulse.t(index2) - pulse.t(index1);
    disp(strcat(pulseCat, "_", lenCat, "_", varCat))
    disp("FWHM is " + string(pulse.FWHM) + " s")
    nGrid = ceil(pulse.FWHM*62.5e6);
    disp("N grid points is " + nGrid)

    % Plot the pulse
    nexttile

    plot(t,p_norm,'LineWidth',LineWidth)
    xlim([0,t_max])
    ylim([-1.5,1.2])
    title(acronym, 'FontSize',FontSize,'FontName',FontName)
    h = gca;
    h.XAxis.Visible = 'off';
    h.YAxis.Visible = 'off';
    h.Color = 'none';
end

% Export the figure
exportgraphics(fig, 'pulses.pdf', 'ContentType', 'vector', 'Resolution', dpi)