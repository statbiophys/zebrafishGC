function [fpr, fnr] = compute_error_rate_adjmat(gRefMat, gcMat)
% function [fpr, fnr] = compute_error_rate_adjmat(gRefMat, gcMat)
%
% compare the GC network gcMat to the groundtruth gRefMat, and 
% output the false positive rate (fpr) and the false negative rate (fnr)
  
nNodes = size(gRefMat,1);

gRefMat = gRefMat - diag(diag(gRefMat));
gcMat = gcMat - diag(diag(gcMat));

gRefMat_flat = gRefMat(:);
gcMat_flat = gcMat(:);
%%
fp = (gcMat_flat == 1)'*(gRefMat_flat == 0) ;
total_tn = sum(gRefMat_flat == 0) - nNodes ;
fpr = fp / total_tn;

fn = (gcMat_flat == 0)'*(gRefMat_flat == 1) ;
total_tp = sum(gRefMat_flat == 1) ;
fnr = fn / total_tp;

end
