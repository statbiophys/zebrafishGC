function [ratio_ipsi] = compute_ipsi_vs_contra(gcMat, nl, nr)
% [ratio_ipsi] = compute_ipsi_vs_contra(gcMat, nl, nr)
%
% compute W_{IC}, the directional bias for ipsilateral links
%
% Input:    1.  gcMat:   the GC matrix
%           2.  nl:      number of neurons on the left chain
%           3.  nr:      number of neurons on the right chain
% Output:       W_{IC} as defined in the manuscript


n = nl + nr;

total_ipsi_left = ...
    sum(gcMat(1:nl, 1:nl) - diag(diag(gcMat(1:nl, 1:nl))), 'all' );

total_ipsi_right = ...
    sum(gcMat(nl+1:n, nl+1:n) - diag(diag(gcMat(nl+1:n, nl+1:n))), 'all' );

total_contra = ...
    sum(gcMat(1:nl, nl+1:n), 'all') + sum(gcMat(nl+1:n, 1:nl), 'all');

n_possible_links_ipsi = nl * (nl - 1) + nr * (nr -1);

n_possible_links_contra = nl * nr * 2;

weighted_total_ipsi = (total_ipsi_left + total_ipsi_right)./ ...
    (n_possible_links_ipsi);

weighted_total_contra = total_contra ./n_possible_links_contra;

ratio_ipsi = weighted_total_ipsi./...
    (weighted_total_ipsi + weighted_total_contra);


end
