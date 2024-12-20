function ax = show_reconstruction_tiledlayout(z,x,img,super_resolved,bubble,showbubbles,c1,c2,n)
% 
% This function visualizes a reconstructed image as part of a tiled layout.
% It supports displaying either diffraction-limited or super-resolved images
% and optionally overlays detected bubbles on the plot.
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
%   n             - Tile index in the tiled layout where the image is displayed.
%
% Returns:
%   ax - Handle to the axis object in the tiled layout.

if super_resolved == false
    fig_title = 'Normal resolution';        % Title for the figure
    colob_title = 'image intensity (dB)';   % Title for the colorbar
    cmap = 'gray';                          % Colormap
else
    cmap = 'gray';
    cmap = flipud(gray);                    % Invert colormap
    fig_title = 'Super-resolved';           % Title for the figure
    colob_title = ['image intensity,'...    % Title for the colorbar
        newline 'linear scale (a.u.)'];
end

% Create the tile and plot
ax = nexttile(n,[3 3]);
imagesc(ax, z.*1e3, x.*1e3, img);

if showbubbles
    % Show the bubbles
    hold(ax, 'on')
    for k = 1:length(bubble)
        plot(ax, bubble(k).z*1e3, bubble(k).x*1e3, 'ro', 'MarkerSize', 3)
    end
end

% Configure axes properties
axis(ax, 'equal')
clim(ax, [c1 c2]);
colormap(ax, cmap);

% Optional: Uncomment if needed
% colob = colorbar(ax);
% ylabel(ax, 'lateral distance [mm]', 'interpreter', 'latex', 'fontsize', 16)
% xlabel(ax, 'axial distance [mm]', 'interpreter', 'latex', 'fontsize', 16)
% ylabel(colob, colob_title, 'interpreter', 'latex', 'fontsize', 8);
% title(ax, fig_title)

% Render the plot
drawnow

% Return the handle to the current axis
ax=gca;
end