function [pulses, pulseSequence] = ...
    getPulseInOutput(pulseProperties, Tfit, TW, dispFig)
% This function performs several steps:
% 1) Gets the input voltages based on the characteristics of the pulses.
% 2) Calculates the transducer pressure response to the input voltages.
% 3) Optional: Plot the transducer input, output and their frequency
% spectrum.
% 4) Adds the input voltage and transducer pressure response to the pulse
% struct.
% 5) Get a list of the pulses that need to be simulated.
%
% [INPUTS]:
%   Tfit:               transfer function of the transducer
%   dispFig:            binary value for figure plotting
%   pulseProperties:    struct with fields:
%   - T_resp:           time which the pulse travels through the medium
%   - NCy_low/med/high: pulse length in cycles for low, med and high freq.
%   - shortChirpDur:    duration of short chirp
%   - longChirpDur:     duration of long chirp
%   - tol:              tolerance value in percentage of the target pulse
%   - calcMod:          binary value for running individual pulses
% [OUTPUT]:
%   pulses:             struct containing normalized pulse information with
%                       voltage, pressure response and pressure gradient
%                       response information
%   pulseSequence       list of pulses that need to be simulated
%
% Rienk Zorgdrager, University of Twente, 2023

T_resp        = pulseProperties.Tresp;
NCy_low       = pulseProperties.Ncy_low;
NCy_med       = pulseProperties.Ncy_med;
NCy_high      = pulseProperties.Ncy_high;
shortChirpDur = pulseProperties.shortChirpDur;
longChirpDur  = pulseProperties.longChirpDur;
tol           = pulseProperties.tol;
calcMod       = pulseProperties.calcMod;

% Initialize the pulse structures
pulses = constructPulseDataframes(pulseProperties);

%% Transducer characteristics
T = Tfit.Tfit;                  % Transfer function
Fs = Tfit.Fs;                   % Sampling frequency

Df = compute_bandwidth(T,Fs);   % Calculate the bandwidth of the transducer
N = length(T);                  % Length of the transducer transfer function
T_IR = 2/Df;                    % Impulse response time

%% Initialize datastructures
NCy = [NCy_low;NCy_med;NCy_high];
chirpDur = [shortChirpDur,longChirpDur];
NCyTrain = NCy_med(2:length(NCy_med));

% Initialize acronyms
acronyms = get_acronyms(calcMod);

%% Loop through the pulses
pulseCat = fieldnames(pulses);

% Loop through all pulse categories
for i = 1:numel(pulseCat)

    if calcMod.(pulseCat{i}) == true % Limit the calculations to the selected pulses

        if (pulseCat{i}) ~= "pulseExpVal" % Do not compute the output parameters for the experimental pulses, we already have them from the Verasonics
            lenCat = fieldnames(pulses.(pulseCat{i}));
            frame = 1;

            % Loop through all length categories of a pulse category (e.g., short vs long)
            for j = 1:numel(lenCat)


                varCat = fieldnames(pulses.(pulseCat{i}).(lenCat{j}));

                % Loop through all variable categories of a length category (e.g. frequency, up-
                % or downsweep)
                for k = 1:numel(varCat)
                    pulse = pulses.(pulseCat{i}).(lenCat{j}).(varCat{k});

                    % Calculate angular center frequency
                    pulse.w = 2*pi*pulse.f0;

                    % Calculate driving time
                    if (pulseCat{i}) == "pulseChirp"
                        pulse.T_drive = chirpDur(j);
                    elseif (pulseCat{i}) == "pulseTristate"
                        pulse.T_drive = (pulse.NCy)/pulse.f0;
                    elseif (pulseCat{i}) == "pulseTrain"
                        pulse.T_drive = NCyTrain(j)*(1/pulse.f0);
                    elseif (lenCat{j}) == "Reference"
                        pulse.T_drive = 1/pulse.f0;
                    else
                        pulse.T_drive = NCy(k,j)*(1/pulse.f0);
                    end

                    % Calculate time vector
                    if (pulseCat{i}) == "pulseTrain"
                        pulse.t = 0:1/Fs:(4*(pulse.T_drive+T_IR) + (pulse.T_drive+T_IR)*sum(pulse.delay)+T_resp);
                    else
                        pulse.t = 0:1/Fs:(pulse.Npulses*pulse.T_drive+T_IR+T_resp);
                    end

                    % Set input voltages
                    if (pulseCat{i}) == "pulseChirp"
                        pulse.V = getChirp(pulse,N);
                    elseif (pulseCat{i}) == "pulseTristate"
                        pulse.V = getTristate(pulse,N);
                    elseif (pulseCat{i}) == "pulseTrain"
                        pulse.V = getPulseTrain(pulse,N,T_IR);
                    elseif (lenCat{j}) == "Reference"
                        pulse = optimize_f0(pulse,T,tol,1);
                    else
                        pulse = optimize_f0(pulse,T,tol,NCy(k,j)); % Optimize the input voltage center frequency to meet the target output center frequency
                    end

                    % Calculate output pressure, pressure gradient, pressure power
                    % spectrum and voltage power spectrum
                    [pulse.p_norm, pulse.dp_norm, pulse.pfft, pulse.Vfft] = getPressure(pulse,T,"voltage");
                    [~,I] = max(abs(pulse.pfft));
                    I = (Fs/length(pulse.pfft))*I;
                    pulse.f_pmax = I;

                    % Calculate FWHM
                    H = abs(hilbert(pulse.p_norm));
                    index1 = find(H >= 0.5,1,"first");
                    index2 = find(H >= 0.5,1,"last");
                    pulse.FWHM = pulse.t(index2) - pulse.t(index1);

                    % Plot the pulse
                    if dispFig == true

                        % Calculate the size of the figure
                        n_fields = zeros(1,numel(lenCat));
                        for m = 1:numel(lenCat)
                            n_fields(m) = length(fieldnames(pulses.(pulseCat{i}).(lenCat{m})));
                        end
                        total_fields = sum(n_fields);

                        plotFunc(pulse, string(pulseCat{i}), string(lenCat{j}), string(varCat{k}), total_fields, frame)
                        frame = frame + 2;
                    end

                    % Add time, input voltage, normalized output pressure, normalized
                    % output pressure gradient, pressure power spectrum and voltage
                    % power spectrum to the pulse struct
                    pulses.(pulseCat{i}).(lenCat{j}).(varCat{k}) = pulse;
                end
            end

        else % Load experimental pulses
            pulses = loadExpPulses(pulses, T, TW, N, Fs);
        end
    end
end

% Get a list of all the pulse types that need to be simulated:
pulseSequence = get_pulse_sequence(pulses,pulseProperties,acronyms);

end

function acronyms = get_acronyms(calcMod)
% Get the acronyms based on the calcMod properties

acronyms = {};
if calcMod.pulseTristate == true
    acronyms{end+1} = 'Tri';
end
if calcMod.pulseSingle == true
    acronyms{end+1} = 'REF';
    acronyms{end+1} = 'S1.7';
    acronyms{end+1} = 'S2.5';
    acronyms{end+1} = 'S3.4';
    acronyms{end+1} = 'L1.7';
    acronyms{end+1} = 'L2.5';
    acronyms{end+1} = 'L3.4';
end
if calcMod.pulseChirp == true
    acronyms{end+1} = 'SUC';
    acronyms{end+1} = 'SDC';
    acronyms{end+1} = 'LUC';
    acronyms{end+1} = 'LDC';
end
if calcMod.pulseTrain == true
    acronyms{end+1} = 'DPT';
end
if calcMod.pulseExpVal == true
    acronyms{end+1} = 'ExpREF';
    acronyms{end+1} = 'ExpSDC';
end
end

function pulses = loadExpPulses(pulses, T, TW, N, Fs)
% Get the experimental pulses
TW = TW.TW;

P_wave = zeros(1,N);

% Obtain the pulses
varCat = fieldnames(pulses.pulseExpVal.Short);

% Loop through all variable categories of a length category (e.g. frequency, up-
% or downsweep)
for k = 1:numel(varCat)
    pulse = pulses.pulseExpVal.Short.(varCat{k});
    
    pulse.t = 0:1/Fs:(length(TW(k).Wvfm1Wy)-1)/Fs;
    pulse.p_norm = P_wave;
    pulse.p_norm(1:length(pulse.t)) = TW(k).Wvfm1Wy';
    [pulse.p_norm, pulse.dp_norm, pulse.pfft, ~] = getPressure(pulse,T,"pressure");
    
    % Obtain center frequency
    [~,I_Pfft] = max(abs(pulse.pfft(1:length(pulse.pfft)/2)));
    f0_Pfft = (pulse.fs/N)*I_Pfft;
    pulse.f0 = f0_Pfft;
    pulse.w = 2*pi*f0_Pfft;

    pulses.pulseExpVal.Short.(varCat{k}) = pulse;
end


end

function pulse = optimize_f0(pulse,T,tol,NCy)
% Optimizes the center frequency of the input pulse such that the output
% pressure pulse has the desired center frequency
% [INPUTS]:
%   pulse:  Pulse struct
%   T:      Transducer transfer function
%   tol:    Permitted error in percentage from the target frequency
%   NCy:    Number of cycles of the desired pulse
% [OUTPUTS]:
%   pulse:  Pulse with updated center frequency of the input voltage

%% Initialize the parameters
P_f0_target = pulse.f0;
eps = tol*P_f0_target;
N = length(T);

pulse.V = getSinglePulse(pulse,N);
[~, ~, Pfft, ~] = getPressure(pulse,T,"voltage");

% Calculate the pressure center frequency
[~,I_Pfft] = max(abs(Pfft(1:length(Pfft)/2)));
f0_Pfft = (pulse.fs/N)*I_Pfft;

% Set-up for the while loop
error = P_f0_target-f0_Pfft;                    % Error from the target freq.
pulse.n_iters = 0;                              % Number of iterations

while abs(error) > eps

    % Update input characteristics
    pulse.f0 = pulse.f0+error;                  % Update the center frequency of the input voltage
    pulse.T_drive = NCy*(1/pulse.f0);           % Calculate new driving time
    pulse.V = getSinglePulse(pulse,N);          % Get the new pulse

    % Calculate the pressure freq spectrum and center frequency
    [~, ~, Pfft, ~] = getPressure(pulse,T,"voltage");     % Calculate the new pressure freq. spectrum

    [~,I_Pfft] = max(abs(Pfft(1:length(Pfft)/2)));
    f0_Pfft = (pulse.fs/N)*I_Pfft;              % Calculate the center frequency

    % Calculate new error
    error = P_f0_target-f0_Pfft;
    pulse.n_iters = pulse.n_iters + 1;
end


end

function V = getChirp(pulse,N)
% Get the chirp voltage with a sweep from f_start to f_end with a driving
% time T_drive at a sampling rate Fs.

% Initialize the parameters
Fs = pulse.fs;
f_start = pulse.f_start;
T_drive = pulse.T_drive;
f_end = pulse.f_end;

% Calculate the driving time vector and the voltage chirp
t = 0:1/Fs:T_drive;
phaseInit = -90;            % Start with a sinusoid
chirpSignal = chirp(t,f_start,T_drive,f_end,'linear',phaseInit);

% Set the voltage vector
V = zeros(1,N);
V(1:length(t)) = chirpSignal;
end

function V = getSinglePulse(pulse,N)
% Get the single pulse voltage with frequency f0 and driving time T_drive
% at a sampling rate Fs.

% Initialize the parameters
f0 = pulse.f0;
T_drive = pulse.T_drive;
Fs = pulse.fs;

% Calculate the driving time vector and the pulse voltage
t = 0:1/Fs:T_drive;
pulseSignal = sin(2*pi*f0.*t);

% Calculate the voltage vector
V = zeros(1,N);
V(1:length(t)) = pulseSignal;
end

function V = getTristate(pulse,N)

% Get a pulse train of alternating positive and negative block pulses, with
% Ncy cycles and frequency f at sampling rate Fs.

% Initialize the paramters
Fs = pulse.fs;
NCy = pulse.NCy;
f = pulse.f0;

ON_Frac = 0.67;             % Fraction of half cycle with high level

NT = Fs/f;                  % Number of sample points per cycle
NT_half = round(Fs/(2*f));  % Number of sample points per half cycle
NT_ON = round(ON_Frac*Fs/(2*f));

V = zeros(1,N);

for k = 1:NCy
    % Positive pulse
    Nstart = 1 + round((k-1)*NT);
    V(Nstart:Nstart+NT_ON) = 1;
    % Negative pulse
    Nstart = Nstart + NT_half;
    V(Nstart:Nstart+NT_ON) = -1;
end

end

function V = getPulseTrain(pulse,N,T_IR)
% Get a pulse train with NPulsesTrain pulses of frequency f0 and duration
% T_drive. The interval between the pulses is T_interval and the polarity
% of the pulses is defined as pol. The delay of the pulse w.r.t. the
% sequence is given in T_delay.

% Initialize the parameters
f0 = pulse.f0;
T_drive = pulse.T_drive;
Fs = pulse.fs;
delays = pulse.delay;
NPulsesTrain = pulse.Npulses;
pol = pulse.pol;


V = zeros(1,N);
t = 0:1/Fs:T_drive;                 % Time vector of individual driving pulse
N_start = 1;
t_pulse = T_drive+T_IR;             % Duration of the pressure pulse
T_delay = t_pulse*delays;           % Delay in seconds

% Loop through the pulses
for i = 1:NPulsesTrain

    % Calculate start and end indices of the input voltage
    N_start = N_start + round(T_delay(i)*Fs);
    N_end = N_start + round(pulse.T_drive*Fs);

    % Set the voltage
    V_pulse = pol(i)*sin(2*pi*f0.*t);
    V(N_start:N_end) = V_pulse;

    % Response time of the transducer
    N_start = N_end + round(T_IR*Fs);

end
end

function [P_norm,dP_norm,Pfft,Vfft] = getPressure(pulse,Tfit,input_qty)
% Calculates the pressure, pressure gradient and power spectra of voltage
% and pressure.
% [Inputs]:
%   pulse:  Struct of a specific pulse
%   Tfit:   Complex double vector representing the transfer function of the transducer
%   input_qty: String with either "voltage" or "pressure"
% [Outputs]:
%   P_norm:      Pressure time domain
%   dP_norm:     Pressure gradient time domain
%   Pfft:   Pressure frequency domain
%   Vfft:   Voltage frequency domain

% Initialize the parameters
Fs = pulse.fs;
M = length(pulse.t);

if input_qty == "voltage"
    % Calculate pressure by convolving with transducer transfer function
    V = pulse.V;
    Vfft = fft(V);
    Pfft = Tfit.*Vfft;        % Fourier transform of pressure signal
else
    Vfft = "n/a";
    Pfft = fft(pulse.p_norm);
end

P = real(ifft(Pfft));

% Apply transfer function to compute derivative
N = length(Pfft);
f = (0:(N-1))/N*Fs;         % Frequency vector
omega = 2*pi*f;             % Angular frequency vector
omega(ceil(N/2+1):N) = -omega(floor(1+N/2):-1:2);
dPfft = 1i*omega.*Pfft;     % Fourier transform of derivative
dP = real(ifft(dPfft));

% Truncate the signal to desired length
P = P(1:M);
dP = dP(1:M);

% Normalise the signal
P0 = max(abs(P));
P_norm = P/P0;
dP_norm = dP/P0;

end

function plotFunc(pulse,pulseName,pulseLen,pulseSpec,nCols,frame)
% Plot the transducer input voltage, transducer output pressure and the
% Fourier transform of the transducer output pressure.

Fs = pulse.fs;

if frame == 1 % Create a new figure for every first frame
    figure('Name',pulseName + " TX input and output",'units','normalized','outerposition',[0 0 1 1]);
end

% Truncate V to the length of the pulse
M = length(pulse.t);
V = pulse.V(1:M);

% Plot the input V and output P
subplot(nCols,2,frame);
yyaxis left
plot(pulse.t*1e6,V);
title(pulseLen +'-'+pulseSpec+' (f0='+pulse.f0/1e6+'MHz)');
ylabel('V');
xlabel('t(\mus)');
yyaxis right;
plot(pulse.t*1e6,pulse.p_norm);
ylabel('P_{normalized}');
ylim([-1,1])

% Plot the frequency spectra
subplot(nCols,2,frame+1);
yyaxis left
plot((Fs/length(pulse.Vfft)*(0:length(pulse.Vfft)-1))/1e6,abs(pulse.Vfft))
ylabel('|fft(V)|')
yyaxis right
plot((Fs/length(pulse.pfft)*(0:length(pulse.pfft)-1))/1e6,abs(pulse.pfft))
title("Magnitude of the complex fft spectrum (f0_{p} = "+ string(pulse.f_pmax/1e6)+"MHz)")
ylabel('|fft(P)|')
xlabel('f (MHz)')
xlim([0, 6])

end

function pulseSequence = get_pulse_sequence(pulses,pulseProperties,acronyms)
% Loop through all the pulse types in pulses and add a pulse type to the
% list if the value of calcMod is true.

% Pulse categories:
pulseCat = fieldnames(pulses);

n = 1; % Sequence counter

for i = 1:numel(pulseCat)

    % Limit the calculations to the selected pulses:
    if pulseProperties.calcMod.(pulseCat{i}) == true

        % Length categories:
        lenCat = fieldnames(pulses.(pulseCat{i}));

        % Loop through all length categories of a pulse category (e.g.,
        % short vs long):
        for j = 1:numel(lenCat)

            % Variable categories:
            varCat = fieldnames(pulses.(pulseCat{i}).(lenCat{j}));

            % Loop through all variable categories of a length category
            % (e.g. frequency, up- or downsweep):
            for k = 1:numel(varCat)

                % Add the pulse type to the list:
                pulseSequence(n).pulseCategory    = pulseCat{i}; %#ok
                pulseSequence(n).lengthCategory   = lenCat{j};   %#ok
                pulseSequence(n).variableCategory = varCat{k};   %#ok
                pulseSequence(n).acronym          = acronyms{n};   %#ok
                n = n + 1;
            end
        end
    end
end

end
