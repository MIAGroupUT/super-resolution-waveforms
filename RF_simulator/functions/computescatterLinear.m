function scattercell = computescatterLinear(bubble,pulse,liquid,gas,shell)
% Calculates the scatter by simulating the linear bubble response and the
% wave propagation to and from the microbubble.
% Nathan Blanken and Rienk Zorgdrager, University of Twente, 2023
   
if pulse.dispProgress
    disp('Solving linear microbubble response ...')
end

% Simulate the wave propagation from transducer to bubble
pulseLocal = pulsePropagation(pulse, liquid, bubble);
pulseLocal.fs = pulse.fs;

[~,~,scatter] = calcBubbleResponseLinear(liquid, ...
    gas, shell, bubble, pulseLocal);

for nbubble = length(bubble):-1:1
    % Store the data in structs, one struct per bubble
    scattercell{nbubble}.ps = scatter.ps(nbubble,:)';
    scattercell{nbubble}.t  = scatter.t(nbubble,:)';
end

end