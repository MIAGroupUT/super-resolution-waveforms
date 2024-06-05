function scattercell = computescatter(bubble,pulse,liquid,gas,shell)
% Calculates the scatter by simulating the bubble response and the wave
% propagation to and from the microbubble.
% Nathan Blanken and Rienk Zorgdrager, University of Twente, 2023

Nb = length(bubble);

batchSize = pulse.batchSize;
Nbatch = ceil(Nb/batchSize);    % Total number of batches

% Write the simulation parameters to cell arrays, each cell
% holding the values for one batch:
scatter_cell  = cell(1,Nbatch);
bubble_cell   = cell(1,Nbatch);
shell_cell    = cell(1,Nbatch);
pulse_cell    = cell(1,Nbatch);

for k = 1:Nbatch

    % Microbubble indices in the current batch
    idx = (k-1)*batchSize + (1:batchSize);
    idx(idx>Nb) = [];

    bubble_cell{k}   = bubble(idx);
    shell_cell{k}    = shell(idx);

    pulse_cell{k}    = pulse;
    pulse_cell{k}.t  = pulse.t(idx,:);
    pulse_cell{k}.tq = pulse.tq(idx,:);
    pulse_cell{k}.p  = pulse.p(idx,:);
    pulse_cell{k}.dp = pulse.dp(idx,:);
end

dispProgress = pulse.dispProgress;

parfor i = 1:Nbatch % Change to parfor later
    
    if dispProgress == true
        disp(['Simulating microbubble batch ' ...
            num2str(i) '/' num2str(Nbatch) ' ...'])
    end

    % Simulate the wave propagation from transducer to bubble
    pulseLocal = pulsePropagation(pulse_cell{i}, liquid, bubble_cell{i});

    % Transform back to retarded time for ODE solver
    pulseTransformed    = pulseLocal;
    pulseTransformed.t  = pulse_cell{i}.t(1,:); % Adjust the time vector
    pulseTransformed.tq = pulse_cell{i}.tq(1,:);

    % Compute the radial response of the bubble
    response = calcBubbleResponse(liquid, ...
        gas, shell_cell{i}, bubble_cell{i}, pulseTransformed);
    
    % Reset time arrays to absolute time   
    for j = 1:size(pulseLocal.t,1)
        response(j).t = pulseLocal.t(j,:)';
    end

    % Do not compute the rapidly decaying r^(-3) term
    nearfield = false;

    % Compute the scattered pressure:
    scatter_cell{i} = calc_scatter(...
        response,liquid,bubble_cell{i},pulse,nearfield);
end

for k = Nbatch:-1:1

    % Microbubble indices in the current batch
    idx = (k-1)*batchSize + (1:batchSize);
    idx(idx>Nb) = [];
    
    % Take the values out of the scatter_cell and store them in an array
    scatterArray(idx,:) = horzcat(scatter_cell{k}.ps);
    timeArray(idx,:)    = horzcat(scatter_cell{k}.t);
end

for nbubble = Nb:-1:1
    % Store the data in structs, one struct per bubble
    scattercell{nbubble}.ps = scatterArray(nbubble,:)';
    scattercell{nbubble}.t = timeArray(nbubble,:)';
end

end