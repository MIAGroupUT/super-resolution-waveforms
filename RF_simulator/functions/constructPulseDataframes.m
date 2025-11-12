function [pulses] = constructPulseDataframes(pulseProperties)
% Construct the dataframes of the different pulses

Fs           = pulseProperties.Fs;
NCyTristate  = pulseProperties.Ncy;
f            = pulseProperties.f;
f0_low       = pulseProperties.f0_init_low;
f0_ref       = pulseProperties.f0_init_ref;
f0_high      = pulseProperties.f0_init_high;
low_f_limit  = pulseProperties.low_f_limit;
high_f_limit = pulseProperties.high_f_limit;
delay        = pulseProperties.delay;
NPulsesTrain = pulseProperties.NCyTrain;

%% Tristate pulse
Tristate = struct('fs', Fs, 'f0', f, 'NCy', NCyTristate,'Npulses',1);
singleTristate = struct('SingleTristate', Tristate);

%% Pulse trains
% Create  alternating pulses
varPol = zeros(1,NPulsesTrain);
for i =1:NPulsesTrain
    if rem(i,2) ~= 0
        varPol(i) = 1;
    elseif rem(i,2) == 0
        varPol(i) = -1;
    end
end
pPol = ones(1,NPulsesTrain);

% Store the values in a struct
Delay    = struct('fs', Fs, 'f0', f0_ref, 'pol', pPol,   'delay',   delay, 'Npulses', NPulsesTrain);
pulseTrainShort = struct('Delay',Delay);

% Chirps
Upsweep   = struct('fs', Fs, 'f_start', low_f_limit,  'f_end', high_f_limit, 'f0', (low_f_limit+high_f_limit)/2, 'Npulses',1);
Downsweep = struct('fs', Fs, 'f_start', high_f_limit, 'f_end', low_f_limit,  'f0', (low_f_limit+high_f_limit)/2, 'Npulses',1);
pulseChirpShort = struct('Upsweep',Upsweep,'Downsweep',Downsweep);
pulseChirpLong  = struct('Upsweep',Upsweep,'Downsweep',Downsweep);

% Single pulses
OneCycle = struct('fs', Fs, 'f0', f0_ref,  'pol', 1, 'delay', 0, 'Npulses', 1);
LowF     = struct('fs', Fs, 'f0', f0_low,  'pol', 1, 'delay', 0, 'Npulses', 1);
MedF     = struct('fs', Fs, 'f0', f0_ref,  'pol', 1, 'delay', 0, 'Npulses', 1);
HighF    = struct('fs', Fs, 'f0', f0_high, 'pol', 1, 'delay', 0, 'Npulses', 1);
pulseSingleRef   = struct('OneCycle',OneCycle);
pulseSingleShort = struct('LowF',LowF,'MedF',MedF,'HighF',HighF);
pulseSingleLong  = struct('LowF',LowF,'MedF',MedF,'HighF',HighF);

% Experimental validation pulses
SIP     = struct('fs', Fs, 'f0', f0_ref, 'pol', 1, 'delay', 0, 'Npulses', 1);
chirp   = struct('fs', Fs, 'f_start', high_f_limit, 'f_end', low_f_limit,  'f0', (low_f_limit+high_f_limit)/2, 'Npulses',1);
pulseExpShort = struct('SIP', SIP, 'chirp', chirp);

% Store all unique pulses by category
pulseTristate = struct('Tristate',  singleTristate);
pulseSingle   = struct('Reference', pulseSingleRef,  'Short', pulseSingleShort, 'Long', pulseSingleLong);
pulseChirp    = struct('Short',     pulseChirpShort, 'Long',  pulseChirpLong);
pulseTrain    = struct('Short',     pulseTrainShort);
pulseExpVal   = struct('Short',     pulseExpShort);

% Create the final struct
pulses = struct('pulseTristate', pulseTristate, 'pulseSingle', pulseSingle, 'pulseChirp', pulseChirp, 'pulseTrain', pulseTrain, 'pulseExpVal', pulseExpVal);

end