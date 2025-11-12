%% SIMULATION INPUT
%% liquid:      liquid properties
% k:            thermal conductivity (W/m/K)
% rho:          density (kg/m^3)
% cp:           specific heat at const. p (J/kg/K)
% nu:           liquid viscosity (Pa.s)
% sig:          surface tension (N/m)
% c:            speed of sound (m/s)
% T0:           initial temperature (K)
% P0:           ambient pressure (Pa)
% Pat:          atmospheric pressure (Pa)
% a,b:          attenuation coef.: beta = a*f0^b in dB/cm, f0 in MHz
% BA:           B/A nonlinearity coefficient
% beta:         nonlinearity parameter (1 + B/A/2)
% ThermalModel  thermal model: 'Isothermal', 'Adiabatic', 'Prosperetti'

%% gas:         gas properties
% k:            thermal conductivity (W/m/K)
% rho:          density (kg/m^3)
% Mg:           molar mass (kg/mol)
% cp:           specific heat at const. p (J/kg/K)
% gam:          heat capacity ratio

%% shell:       shell properties
% model         'Marmottant', 'Segers', or 'SegersTable'
% Ks            shell viscosity (N.s/m)
% sig_0:        initial surface tension (N/m)
% chi:          shell stifness (N/m) (Marmottant model)
% Rb:           buckling radius (m) (Marmottant model)
% sig_l:        surface tension of surrounding liquid
% A_N:          reference surface area (m^2) (Segers model)

%% pulses:      drive pulse properties
% f0:           centre frequency (Hz)
% w:            centre frequency (angular) (Hz)
% Ncy:          number of cycles 
% Npulses:      number of pulses in a sequence
% pol:          polarity of the pulse, positive (1) or negative (-1)
% delay:        delay between pulses (fraction of total pulse duration)
% f_start:      start of frequency sweep (Hz)
% f_end:        end of frequency sweep (Hz)
% PA:           pressure amplitude (Pa)
% fs:           sample frequency (Hz)
% t:            time vector (s)
% tq:           query time vector (s)
% p:            pressure vector (Pa)
% p_norm:       normalized pressure vector
% dp:           time derivate pressure vector (Pa/s)
% dp_norm:      normalized time derivate pressure vector
% dispProgress: flag to display the simulation progress
% batchSize:    number of files in a batch for parallelized solving

%% bubble:      bubble properties of i-th bubble
% z:            distance from transducer (m)
% R0:           initial radius bubble (m)

%% COMPUTED EQUATION PARAMETERS
%% eqparam:     equation parameters 

% nu_th:        thermal damping constant (Pa.s)
% nu_rad:       radiative damping constant (Pa.s)
% nu_vis:       viscous damping constant (liquid viscosity) (Pa.s)
% nu:           total damping constant (Pa.s)
% Ks:           shell dilatational viscosity (N.s/m)
% kappa:        polytropic exponent


%% SIMULATION OUTPUT
%% response:    radial response bubble
% R:            radial response vector (m)
% Rdot:         time derivative radial response (radial velocity) (m/s)
% t:            time vector (s)

%% scatter:     scattered pressure
% ps:           scattered pressure vector (Pa)
% t:            time vector (s)

