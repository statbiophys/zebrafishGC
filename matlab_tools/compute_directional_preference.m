function [ratio_htt] = compute_directional_preference(gcMat, nl, nr)
% [ratio_htt] = compute_directional_preference(gcMat, nl, nr)
% htt stands for "head-to-tail", or rostral-to-caudal.
%
% compute W_{RC}, the directional bias for ipsilateral links
%
% Input:    1.  gcMat:   the GC matrix
%           2.  nl:      number of neurons on the left chain
%           3.  nr:      number of neurons on the right chain
% Output:       W_{RC} as defined in the manuscript


%% 

n = nl + nr;
% Version 1. only care about direction. Node 1 drive node 5 is okay.

gcMat_left = gcMat(1:nl, 1:nl);

gc_tth_left = makeUpt(gcMat_left); % need to check
gc_htt_left = makeUpt(gcMat_left');

%%
gcMat_right = gcMat(nl+1:n, nl+1:n);
gc_tth_right = makeUpt(gcMat_right); % need to check
gc_htt_right = makeUpt(gcMat_right');

%%
total_weight_htt = sum(gc_htt_left) + sum(gc_htt_right);
total_weight_tth = sum(gc_tth_left) + sum(gc_tth_right);

ratio_htt = total_weight_htt./(total_weight_tth + total_weight_htt);


end




