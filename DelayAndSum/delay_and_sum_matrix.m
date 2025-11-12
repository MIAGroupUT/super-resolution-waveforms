function img = delay_and_sum_matrix(RF_data, M_DAS, apod, Nx_rec, Nz_rec)
% Matrix based delay-and-sum reconstruction, 2024\09\27, written by
% Guillaume Lajoinie and adjusted by Rienk Zorgdrager
%
% INPUTS:
%   RF_data:  matrix of RF data
%   M_DAS:    matrix representing element contributions to every pixel in the image
%   apod:     matrix representing the apodization values for every pixel in the image
%   Nx_rec:   number of pixels in lateral direction
%   Nz_rec:   number of pixels in axial direction

%
% OUTPUTS:
%   img:      image

%% COMPUTE APODIZATION AND DELAYS FOR EVERY ELEMENT IN THE IMAGE

apod(M_DAS>numel(RF_data(:,:,1))) = 0;  % Put apodization to zero if it exceeds the signal length
M_DAS(M_DAS>numel(RF_data(:,:,1))) = 1; % Put delay to one if it exceeds the signal length

% Normalization factor
norm_v = sum(apod,3);

%% Reconstruct the image
% Matrix to store the reconstructed image pixels:
img = zeros(Nx_rec,Nz_rec,size(RF_data,3));

% Loop through the frame datastructure
for ifr  = 1:size(RF_data,3)
    
    % Get the current frame
    RF_loc = RF_data(:,:,ifr);
    RF_loc = reshape(RF_loc',[numel(RF_loc),1] );
    
    % Compute the pixel intensities
    IMGloc = RF_loc(M_DAS).*apod;
    IMGloc = sum(IMGloc,3)./norm_v;
    IMGloc(isnan(IMGloc)) = 0;
    
    % Assign intensities to pixels in image
    img(:,:,ifr) = IMGloc;
    
end

end