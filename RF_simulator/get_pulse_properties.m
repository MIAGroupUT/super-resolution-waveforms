function pulseProperties = get_pulse_properties()
% List of all pulse properties

% Tristate pulse
P.Ncy = 1;                    % Total number of cycles
P.f = 1.7e6;                  % Centre frequency (Hz)

% Single pulse
P.f0_init_low = 1.7e6;        % Lower frequency (experimentally derived to get p0 to +- 1.7MHz)
P.f0_init_ref = 2.5e6;        % Middle frequency - Reference frequency (experimentally derived to get p0 to +- 2.5MHz)
P.f0_init_high = 3.4e6;       % Higher frequency (experimentally derived to get p0 to +- 3.4MHz)
P.Ncy_low = [0,3,6];          % Number of cycles of lower frequency pulse as [ref,short,long]
P.Ncy_med = [1,4.5,9];        % Number of cycles of middle frequency pulse as [ref,short,long]
P.Ncy_high = [0,6,12];        % Number of cycles of higher frequency pulse as [ref,short,long]
P.tol = 0.01;                 % Tolerance of deviation from the pressure center frequency

% Chirp
P.low_f_limit = 1.2e6;        % Minimum chirp frequency
P.high_f_limit = 4.0e6;       % Maximum chirp frequency
P.shortChirpDur = 5e-6;       % Short chirp duration in s
P.longChirpDur = ...
    3*P.shortChirpDur;        % Long chirp duration in s

% Pulse train (T.B.D.)
P.delay = [0,0.3742,1.6947,0.3191];     % Delay between pulses as a fraction of pulse duration
P.NCyTrain = 4;               % Number of pulses in pulse train

% Generic pulse properties
P.Tresp = 4e-6;               % Echo receive time after pulse (s)
P.Pmin = 5;                   % Minimum acoustic pressure (kPa)
P.Pmax = 250;                 % Maximum acoustic pressure (kPa)

% Set calculation modes
% Define which pulses need to be simulated
P.calcMod = struct( ...
    'pulseTristate',false, ...
    'pulseSingle',true, ...
    'pulseChirp',true, ...
    'pulseTrain',true ...
    );

pulseProperties = P;

end