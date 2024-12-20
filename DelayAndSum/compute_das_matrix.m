function [M_DAS, apod] = compute_das_matrix(t, x, z, x_el, c, Fs, focus, NDt, interp_fun)
% COMPUTE_DAS_MATRIX Compute delay-and-sum reconstruction matrix. Construct
% a sparse matrix that performs delay-and-sum reconstruction on element RF
% data. The matrix is M-by-N, where M is the total number of pixels (Nx*Ny) 
% and N is the total number of elements in the RF data (Nt*Nelem).
%
% INPUT ARGUMENTS:
%   t:            time array corresponding to the RF data (column array)[s]
%   x:            lateral coordinates of the reconstruction [m]
%   z:            axial   coordinates of the reconstruction [m]
%   x_el:         transducer element coordinates [m]
%   c:            speed of sound for the reconstruction [m/s]
%   Fs:           sampling rate of the RF data [Hz]
%   focus:        lateral focus of the transmit beam
%   NDt:          total duration for which needs to be corrected [samples]
%   interp_fun:   interpolated sensitivity curve of the transducer elements
%
% OUTPUTS:
%   M_DAS:        Matrix containing the indices of the RF intensities
%                   contributing to one pixel
%   apod:         Matrix containing the apodization values for every point
%                   in the RF signal for every element

%% Set angle limit
alpha_th = pi/4; % Threshold angle for apodization

Nx      = length(x);     % Number of pixels in x
Nz      = length(z);     % Number of pixels in z
Nt      = length(t);     % Number of time samples per element
Nelem   = length(x_el);  % Number of elements

% Image coordinate grid:
[X, Z] = ndgrid(x, z);

if isfinite(focus) && focus >=0
    error('Only negative focus and infinite focus supported.')
end

X = repmat(X, [1,1,Nelem]);
Z = repmat(Z, [1,1,Nelem]);
X_el = zeros(1,1,Nelem);
X_el(1,1,:) = x_el;
X_el = repmat(X_el, [Nx,Nz,1]);

%% COMPUTE THE TIME DELAY FOR EVERY POINT IN SPACE FOR EVERY ELEMENT
% This is the arrival time in space at which an element receiver the
% signal.

F = abs(focus);

if isfinite(F)
    % Focused beam (not tested):
    % t_del = sqrt((X-X_el).^2 + Z.^2)/c + sqrt((Z+F).^2 + X.^2)/c - F/c; %
else
    % Plane wave:
    t_del = sqrt((X-X_el).^2 + Z.^2)./c + Z./c;
end

% receive time from each point in space for every element
t_del = t_del+NDt/Fs;

% Convert to samples
M_DAS = round(t_del.*Fs);

% Give the DAS matrix a unique index number. This helps to locate the RF
% lines when they are reshaped into a long vector in the previous code.
cov = (0:1:Nelem-1).*Nt;
covM = zeros(1,1,Nelem);
covM(1,1,:) = cov;
covM = repmat(covM, [size(X,1),size(X,2),1]);

M_DAS = M_DAS+covM;

%% COMPUTE APODIZATION
% Angle (f-number) dependent apodization (discard elements with a greater 
% angle to the pixel than the threshold angle):
alpha = atan((X_el-X)./Z);

% Compute apodization using the sensitivity curve for different angles
alpha2 = reshape(alpha,[numel(alpha) 1]);
apod = interp_fun(alpha2);
apod = reshape(apod,[size(alpha,1) size(alpha,2) size(alpha,3)]);
clear alpha2

% Use apodization to correct for angle sensitivity
apod(apod<=0) = nan;
apod = 1./apod; 
apod(isnan(apod)) = 0;

% Remove larger angles to avoid artefacts
apod(abs(alpha) > alpha_th) = 0;

end