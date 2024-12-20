function ax = show_reconstruction(z,x,img,super_resolved,bubble,showbubbles,c1,c2)
%
% This function visualizes a reconstructed image and optionally overlays
% detected bubbles. The appearance of the image depends on whether it is 
% super-resolved or diffraction-limited.
%
% Parameters:
%   z             - Axial coordinates of the image [in meters].
%   x             - Lateral coordinates of the image [in meters].
%   img           - Image matrix to display (e.g., intensity values).
%   super_resolved - Boolean flag; true for super-resolved images, false otherwise.
%   bubble        - Struct array containing bubble positions, with fields:
%                   * z: Axial positions of bubbles [in meters].
%                   * x: Lateral positions of bubbles [in meters].
%   showbubbles   - Boolean flag; true to overlay bubble positions, false otherwise.
%   c1, c2        - Intensity range for the color scale (used for `clim`).
%
% Returns:
%   ax - Handle to the current axis object.

%% IMAGE SETTINGS
% fontSize = 12;
fontSize = 10;
fontName = 'Times New Roman';
% fontName = 'Arial';

%% COLORBAR/MAP SETTINGS
if super_resolved == false
    fig_title = 'Diffraction-limited';        % Title for the figure
    colob_title = 'image intensity (dB)';   % Title for the colorbar
    cmap = 'gray';                          % Colormap

else
    cmap = 'gray';
    cmap = colormap(cmap);
    cmap = colormap(flipud(cmap));          % Invert colormap

    fig_title = 'Super-resolved';           % Title for the figure
    colob_title = ['image intensity,'...    % Title for the colorbar
        newline 'linear scale (a.u.)'];
end

%% SHOW THE IMAGE
imagesc(z.*1e3,x.*1e3,img);

%% SHOW THE BUBBLES
if showbubbles
    hold on

    for k = 1:length(bubble)
        plot(bubble(k).z*1e3, bubble(k).x*1e3, 'ro')
    end

end

%% FORMAT THE AXIS
clim([c1 c2]);

% Define colorbar location
% colob = colorbar('southoutside');
% ylabel(colob,colob_title,'interpreter', 'latex','fontsize',fontSize, 'FontName',fontName);

% Axis formatting
ylabel('lateral distance [mm]','interpreter', 'latex','fontsize',fontSize, 'FontName',fontName)
xlabel('axial distance [mm]','interpreter', 'latex','fontsize',fontSize, 'FontName',fontName)
title(fig_title, 'FontSize', fontSize + 2)
axis equal
ylim([min(x)*1e3, max(x)*1e3])
xlim([min(z)*1e3, max(z)*1e3])
drawnow
colormap(cmap);
ax = gca;
end
