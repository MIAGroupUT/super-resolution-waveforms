% Write arrays from RF simulation .mat files to a .txt file.
% Two column version (acoustic pressure field constant)
%
% Column 1: bubble count (number of bubbles at each sample point)
% Column 2: RF voltage values
% Column 3: pressure field distribution
%
% Nathan Blanken, University of Twente, 2020
% Modified by Rienk Zorgdrager, University of Twente, 2024

delim = "\";
NTRN = 3000;    % Number of training data files
NVAL = 1000;    % Number of validation data files

% source directory:
parent_src = "D:\SRML-1D-pulse-types\Results\RF signals\mat_files";

% destination directory
parent_dst = "D:\SRML-1D-pulse-types\Results\RF signals\txt_files";

% descriptives directory
descr_dir = "D:\SRML-1D-pulse-types\Results\RF signals\mat_files\simulationDescriptives\05-March-2024_sim1-5000.mat"; % Adjust to actual name

% List pulses in source directory
pulselist = dir(parent_src);
pulselist = pulselist(3:end);
pulse_filter = contains({pulselist.name},'compressed');
pulselist = pulselist(pulse_filter);

Npulses = length(pulselist);

% Load simulation descriptives
load(descr_dir)

%% Loop over the pulses
for p = 1:Npulses

    % Create pulse parent directories
    pulsedir_src = parent_src + delim + string(pulselist(p).name);
    pulsedir_dst = parent_dst + delim + string(pulselist(p).name);

    % List files in pulse directory
    filelist = dir(pulsedir_src);
    filelist = filelist(3:end);

    Nfiles = length(filelist);

    %% Loop over the files

    % Temporary if-statement for compression only
    if contains(pulselist(p).name,'compressed')==1
        for n = 1:Nfiles

            %clc
            disp(n)

            % Source file
            filename_src = filelist(n).name;
            filenumber = str2double(filename_src(7:end-4));

            % Create name destination file
            filename_dst = strcat(filename_src(1:end-4),'.txt');

            % Load data
            load(strcat(pulsedir_src,delim,filename_src))

            % Transform data
            RFvoltage = RF.V;
            bubbleCount = getBubbleCount(bubble,RF,domain);
            % pressureField = pfield.PA;  % (in Pa)

            Nb = length(bubble);   % Total number of bubbles
            %PA = p.A/1e3;       % Acoustic pressure (kPa)

            % Sort data in training, validation, and test data:
            subfolder = delim + "TRAINING";
            if filenumber > NTRN
                subfolder = delim + "VALIDATION";
            end
            if filenumber > NTRN + NVAL
                subfolder = delim + "TESTING";
            end

            % Write data to text file
            A = [bubbleCount; RFvoltage];
            %A = [bubbleCount; RFvoltage; pressureField];

            if ~exist(strcat(pulsedir_dst,subfolder)) == 1
                mkdir(strcat(pulsedir_dst,subfolder))
            end

            fileID = fopen(strcat(pulsedir_dst,subfolder,delim,filename_dst),'w');

            % Write header:
            fprintf(fileID,['"Generated with writeResultsToTxt.m from ' ...
                filename_src '"\n']);

            fprintf(fileID,'"Number of bubbles:",%d\n',Nb);
            %fprintf(fileID,'"Acoustic pressure (kPa):",%.2f\n',PA);

            fprintf(fileID,['"Bubble count","Voltage (a.u)"\n']);

            % Write data:
            fprintf(fileID,'%d,%.10f\n',A);
            fclose(fileID);
        end
    end

    %% Write the simulation descriptives
    filename_descr = "simulationDescriptives.txt";

    % Training dataset
    pulsedir_dst_trn = pulsedir_dst + delim + "TRAINING" + delim + filename_descr;
    descriptivesToTXT(pulsedir_dst_trn,simTable(1:NTRN,:),pulselist(p).name)

    % Validation dataset
    pulsedir_dst_val = pulsedir_dst + delim + "VALIDATION" + delim + filename_descr;
    descriptivesToTXT(pulsedir_dst_val,simTable((NTRN+1):(NTRN+NVAL),:),pulselist(p).name)

    % Test dataset
    pulsedir_dst_tst = pulsedir_dst + delim + "TESTING" + delim + filename_descr;
    descriptivesToTXT(pulsedir_dst_tst,simTable((NTRN+NVAL+1):end,:),pulselist(p).name)
end

function bubbleCount = getBubbleCount(bubble,RF,domain)
% Get number of bubbles at each sample point

Nb = length(bubble);        % number of bubbles
N  = length(RF(1).V);          % number of sample points

% Get arrays of bubble locations
z = [bubble.z];             % Axial coordinates (m)
x = [bubble.x];             % Lateral coordinates (m)
r = sqrt(x.^2 + z.^2);      % Distance from centre element (m)

% Convert to RF sample number
t = (z+r)/domain.c;         % Echo arrival time (s)
I = t*RF.fs;                % Convert to sample number
I = round(I)+1;

T = 2*domain.depth/domain.c;    % Total receive time
RF.t = 0:1/RF.fs:T;          	% Time vector

% Construct bubble count array
bubbleCount = zeros(1,N);
for n = 1:Nb
    bubbleCount(I(n)) = bubbleCount(I(n)) + 1;
end

end

function simulationDescriptives = descriptivesToTXT(savepath, table, pulsename)
% Store the number of bubbles, emitted pressure, computer and simulation
% time in a .txt file.

% Select the pulse specific elapsed time
pulsename = erase(pulsename,"_");

if contains(pulsename,"compressed")==false
    timevar = "t_"+ string(pulsename);
else
    timevar = "t_"+ erase(string(pulsename),"compressed");
end

% write data to .txt file
Nb          = table.Nb;
PA          = table.PA;
computer    = table.computer;
t_tot       = table.t_tot;
t           = table.(timevar);

% open fileID
fileID = fopen(savepath,'w');

% write header
fprintf(fileID,['"Generated with writeResultsToTxt.m from ' ...
    'simulationDescriptives.mat"\n']);

fprintf(fileID,['"Nb","PA","computer","t_tot","t_pulse"\n']);

% Write data
fmt = ['%d,%.10f,%s,%.3f,%.3f\n'];

for i=1:size(Nb,1)
    fprintf(fileID,fmt,Nb(i), PA(i), computer(i), t_tot(i), t(i));
end

% Close file
fclose(fileID);

end