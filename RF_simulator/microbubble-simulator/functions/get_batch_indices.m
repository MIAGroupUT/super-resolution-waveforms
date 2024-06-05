function idx = get_batch_indices(batchIndex, N_MB, batchSize, radii)
% Sort the bubbles by size and group them into batches of batchSize.
% Bubbles of similar size have similar characteristic timescales. Having
% similarly sized microbubbles in a batch is expected to speed up
% computation.

% Linearly increasing indices:
idxLinear = (batchIndex-1)*batchSize + (1:batchSize);
idxLinear(idxLinear>N_MB) = [];

% Microbubble indices sorted by size:
[~,idxSort] = sort(radii);
idx = idxSort(idxLinear);
end