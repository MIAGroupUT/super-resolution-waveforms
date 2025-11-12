function img = delay_and_sum(RF_data, IM_WI, IM_DE, pix_siz, ...
    x_el, c, sig_dur, Fs, sens, angle_correction)
% Delay and sum reconstruction
%
% RF_data:  matrix of RF data
% IM_WI:    image width (m)
% IM_DE:    image depth (m)
% pix_siz:  image pixel size (m)
% x_el:     transducer element coordinates (m)
% c:        speed of sound (m/s)
% sig_dur:  signal duration (s)
% Fs:       sampling frequency
% sens:     receiver sensitivity

% Grid of the new image
% Define here the vector defining the x and y axis of the new image and the angles in the sensitivity vector.
x_rec = -IM_WI/2:pix_siz:IM_WI/2;
z_rec = 0:pix_siz:IM_DE;
angles  = (-pi/2:pi/100:pi/2);          % Angle

% Interpolate the sensitivity of the transducer
angles_interp = (-pi/2:pi/1000:pi/2);          % Angle
sens_interp = interp1(angles, sens, angles_interp);

Nx_rec = length(x_rec);         % number of pixels
Nz_rec = length(z_rec);         % number of pixels
Nelem  = length(x_el);        	% number of elements
N      = size(RF_data,2);       % number of samples per RF line

% Add zeros to represent time delays out of range:
RF_data = [RF_data zeros(Nelem,1)];

% Matrix to store the reconstructed image pixels:
img = zeros(Nx_rec,Nz_rec);

for i = 1:size(img,1)
    % Image rows

    % Display progress:
    clc
    recon_progress = floor((i-1)/(size(img,1)-1)*1000)/10;
    %disp([num2str(recon_progress) ' % of image reconstructed']);
    fprintf('%0.1f %% of image reconstructed\n', recon_progress);

    for j=1:size(img,2)
        % Image columns

        X = x_rec(i);               % target point x-coordinate
        Z = z_rec(j);               % target point z-coordinate

        % Time delay for point (X,Y)
        t_del = Z/c + sqrt((X-x_el).^2 + Z^2)/c;

        %% Adjust this
        if angle_correction == true
            th          = atan((x_el-X)/pix_siz);     % Compute the of the RF inputs to the pixel
            sens_list   = getSensitivityList(th,angles_interp,sens_interp); % Get the sensitivity of the pixel to every RF line
        end

        % Corresponding sample points:
        Ntimedel = round(t_del*Fs + sig_dur);
        % NOTE: for a linear scatterer, we should add sig_dur/2 to the time
        % delay. However, it appears that the maximum in the scattered
        % pressure of the bubble lags behind the maximum in the drive
        % pulse. Adding sig_dur seems to work pretty well.

        % Time delays out of range do not contribute:
            Ntimedel(Ntimedel > N) = N+1;
        Ntimedel(Ntimedel < 1) = N+1;

        divby = length(Ntimedel)-sum(Ntimedel==N+1); % Number of elements which contribute

        % Add signal contributions per element
        val = 0;

        for g = 1:Nelem
            if angle_correction == true
                
                val = val+RF_data(g,Ntimedel(g))/sens_list(g);
            else
                val = val+RF_data(g,Ntimedel(g));
            end
        end

        img(i,j) = val/divby;

    end
end

end

function sens_list = getSensitivityList(th,angles,sens)

idx_list = zeros(1,length(th));
sens_list = zeros(1,length(th));

% Find the closest value to the angle in angles.
for a = 1:length(th)

    [~,idx]         = min(abs(angles-th(a)));
    idx_list(a)     = idx;
    sens_list(a)    = sens(idx);
end

end