%% RF line simulator
%
% Simulates the element RF echo data coming from a homogeneous distribution
% of microbubbles in a rectangular domain. The simulator simulates a plane,
% homogeneous transmit wave. Bubble responses are computed with a
% Marmottant-type Rayleigh-Plesset equation, which takes viscous,
% radiation, shell, and thermal damping into account. Bubble-bubble
% interactions are neglected. The pulse shape is based on the pulse from
% the P4-1 transducer.
%
% Nathan Blanken and Rienk Zorgdrager, University of Twente, 2023

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZE

clear
close all

dispFig = false;            % Display figures
dispProgress = false;       % Show ODE solver progress
single_element = true;      % Only compute RF data centre element

delim = '\';            	% OS specific directory delimiter
filedir = ...               % Save directory for simulation results
    'D:\SRML-1D-pulse-types\Results\RF signals\mat_files';

rng('shuffle')                      % Shuffle the random number generator

% Add the functions folder to path
addpath './functions'
addpath './microbubble-simulator'
addpath './microbubble-simulator/functions'

% Get the transducer transfer functions of the P4-1 transducer
Tfit = load('TransmitTransferFunctionFit.mat');
Hfit = load('ReceiveTransferFunctionFit.mat','Hfit');

linearsimulation = false; % Set to true for a linear bubble response

%% Pulse properties

pulseProperties = get_pulse_properties();

% Simulation and recording sampling rate:
Fs = Tfit.Fs;              	% Simulation sampling rate (Hz)
downsample = 4;             % Measurement sampling rate: Fs/downsample
pulseProperties.downsample = downsample;
pulseProperties.Fs = Fs;

% Acoustic pressure range:
Pmin = pulseProperties.Pmin; % Minimum acoustic pressure (kPa)
Pmax = pulseProperties.Pmax; % Maximum acoustic pressure (kPa)

%% Material properties and environmental conditions
[liquid, gas] = getMaterialProperties();

% Select a thermal model: 'Adiabatic', 'Isothermal', or 'Propsperetti':
liquid.ThermalModel = 'Prosperetti';

if linearsimulation == true
    liquid.beta = 0;
end

%% Scan domain and transducer properties
depth = 0.1;                % Imaging depth (m)
width = 0.028;              % Transducer width (m) (P4-1 transducer)
c = liquid.c;             	% Speed of sound in the medium (m/s)
d1 = 0.0037;                % left boundary bubble locations (m)
d2 = depth - 0.015;         % right boundary bubble locations (m)
N_elem = 96;                % Number of transducer elements

% Element positions (m):
if single_element == true
    x_el = 0;
else
    x_el = linspace(-width/2,width/2,N_elem);
end

domain = struct('depth',depth,'width',width,'c',c,'d1',d1,'d2',d2,...
    'x_el',x_el);
clear depth width c d1 d2 x_el N_elem

%% Bubble properties
% Typical value for polydispersity index: PDI = 5% (Segers et al, Soft
% Matter, 14, 2018).
Nmax = 1000;                % Maximum number of bubbles
Nmin = 10;               	% Minimum number of bubbles
R0 = 2.4;                   % Mean bubble radius (um)
sR0 = R0*0.05;           	% Standard dev. bubble radii (um)
r0 = 0.001;                 % Scattered pressure sensor distance to centre
shell_model = 'Segers';     % Marmottant, Segers, or SegersTable
sig_0 = 10e-3;              % Equilibrium surface tension bubble (N/m).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIMULATE THE RF SIGNALS
% Calculate the driving time, input voltage and normalized pressure outcome
[pulses, pulseSequence] = getPulseInOutput(pulseProperties, Tfit, dispFig);

% Simulation settings
batchSize = 11;
NSIM_start = 1;          % Start the simulations at this number
NSIM_end = 5000;         % Stop the simulations at this number

sim = struct("Nb",[],"PA",[],"t_tot",[],"computer",[]);

for n = NSIM_start:NSIM_end
    
    tStartSim = tic;

    %% Calculate pressure
    % Pulse with a randomly allocated acoustic pressure    
    PA = (Pmax-Pmin)*rand() + Pmin;     % Acoustic pressure amplitude (kPa)
    PA = PA*1000;                       % Acoustic pressure amplitude (Pa)

    %% Get random bubbles
    Nb = round((Nmax - Nmin)*rand() + Nmin);    % Number of bubbles
    [bubble, shell] = getBubbles(Nb,R0,sR0,...
        domain,liquid,shell_model,sig_0,r0);

    %% Display simulation status
    disp("Running sample: " + string(n) +", Number of microbubbles: " + ...
        string(Nb) +  ", PA: " + string(PA))

    % Loop through all the pulse types:
    for npulse = 1:length(pulseSequence)

        pulseCat = pulseSequence(npulse).pulseCategory;
        lenCat   = pulseSequence(npulse).lengthCategory;
        varCat   = pulseSequence(npulse).variableCategory;   

        % Make a folder for the results of each pulse variant
        var_dir = strcat(filedir, delim, string(pulseCat) + "_" ...
            + string(lenCat) + "_" + string(varCat));
        if ~exist(var_dir,'dir')
            mkdir(var_dir)
        end

        tPulseStart = tic;

        % Load the pulse
        pulse = pulses.(pulseCat).(lenCat).(varCat);
        
        % Calculate pressure amplitudes of pulse       
        pulse.p  = repmat(pulse.p_norm * PA,Nb,1);
        pulse.dp = repmat(pulse.dp_norm * PA,Nb,1);
        pulse.t  = repmat(pulse.t,Nb,1);

        % Adjust names of variables for running the microbubble simulator
        % modules
        pulse.f = pulse.f0;
        pulse.tq = pulse.t;             % Query times for ODE solver
        pulse.dispProgress = dispProgress;
        pulse.batchSize = batchSize;    % Batch size for the ODE solver

        % Compute the response from the cloud of microbubbles:
        if linearsimulation == true
            scatter = computescatterLinear(bubble,pulse,liquid,gas,shell);
        else
            scatter = computescatter(bubble,pulse,liquid,gas,shell);
        end
        
        %% Wave propgation and RF signal construction
        RF = constructRF(scatter,bubble,liquid,pulse,domain,Fs, ...
            downsample,Hfit.Hfit);
        
        tPulseEnd = toc(tPulseStart);
        
        sim(n).("t_" + string(pulseCat)+string(lenCat)+string(varCat)) ...
            = tPulseEnd;

        disp(string(pulseCat) + " - " + string(lenCat) + string(varCat) ...
            + " took " + string(tPulseEnd) + "sec")
        
        %% Save results
        
        % Remove redundant fields to save disk space:
        RF = rmfield(RF,'t');
        pulse = rmfield(pulse,'dp');
        pulse = rmfield(pulse,'t');

        filename = sprintf("RFDATA%05.0f", n);
        save(strcat(var_dir,delim,filename,'.mat'),...
            'domain', 'liquid','gas','shell','pulse',...
            'bubble','RF')
        
    end
    
    %% Store simulation parameters for evaluation
    tEndSim = toc(tStartSim);

    sim(n).t_tot = tEndSim;
    sim(n).Nb = Nb;
    sim(n).PA = PA;
    sim(n).computer = "Maroilles";
end

%% Store simulation descriptives
descr_dir = strcat(filedir, delim,"simulationDescriptives");

if ~exist(descr_dir,'dir')
    mkdir(descr_dir)
end

save(strcat(descr_dir,delim, string(datetime("today")), "_sim",string(NSIM_start),"-",string(NSIM_end)),"sim")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the last simulated RF line and clear variables

figure

T = 2*domain.depth/domain.c;    % Total receive time
RF(1).t = 0:1/RF(1).fs:T;   	% time vector

plot(RF(1).t,RF(1).V)
hold on
xlabel('t (s)')
ylabel('V (V)')

clear NSIM dispFig delim filedir Tfit Hfit
clear Fs downsample Pmin Pmax
clear Nmax Nmin R0 sR0 r0 shell_model sig_0
clear PA Nb PAlocal pulseLocal N_elem RFc scatter_elem scatter pulse
clear n M
clear filename
clear T z x_el scattercell single_element
