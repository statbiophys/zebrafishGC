function [fstat, pvalue, rss_reduced, rss_full] = ... %, reduced_params, full_params] = ...
    measure_gc_multivariate(x1, x2, xrest, lags, plotFlag, constantFlag)
% [fstat, pvalue, rss_reduced, rss_full] = ...
%    measure_gc_multivariate(x1, x2, xrest, lags, ...
%    plotFlag, constantFlag)
%  
%  Measure multivariate granger causality from x2 to x1, conditioned on xrest
%
%  Input: 1. x1: one-dimensional time-series of the "receiver" neuron
%         2. x2: one-dimensional time-series of the "drive" neuron
%         3. xrest: the conditional variables, can be multidimensional time
%                   series
%         4. lags: scalar, the time lags, or the model order.
%         5. plotFlag (default 0):        if 1, plot figures.
%         6. constantFlag (default 1):    if 1, the linear regression will
%                                               estimate the constant
%                                         if 0, the constant is set to 0.
%
%  Output: 1. fstat: f-statistics of the GC analysis
%          2. pvalue: the corresponding pvalue of the fstat compared to an
%          f-distribution
%          3. rss_reduced: the sum of squared residue for the reduced model
%          4. rss_full:    the sum of squared residue for the full model

if nargin < 5
    plotFlag = 0;
end

if nargin < 6
    constantFlag = 1;
end


nSamples = length(x1);
nSamples_regress = nSamples - lags;

if constantFlag
    y1_reduced = [ones(1,length(x1)) ; hankel([zeros(1,lags-1) x1(1)],x1)];
else
    y1_reduced = hankel([zeros(1,lags-1) x1(1)], x1);
end

nNodes_rest = size(xrest,1);
for iNode = 1:nNodes_rest
    y1_reduced = [y1_reduced ; ...
        hankel([zeros(1,lags-1) xrest(iNode,1)],xrest(iNode,:))];
end


y1_full = [y1_reduced; hankel([zeros(1,lags-1) x2(1)],x2)];

y1_reduced = y1_reduced(:,lags:end-1); 
y1_full = y1_full(:,lags:end-1);

x1_lagcut = x1(1+lags:end);

reduced_params = lsqminnorm(y1_reduced',x1_lagcut');
full_params = lsqminnorm(y1_full',x1_lagcut');

residue_reduced = x1_lagcut' - y1_reduced' * reduced_params;
residue_full = x1_lagcut' - y1_full' * full_params;

%%
rss_reduced = sum(residue_reduced.^2);
rss_full = sum(residue_full.^2);

if constantFlag
     fstat = (rss_reduced/rss_full - 1) ...
            * (nSamples_regress - (nNodes_rest+2) * lags - 1)/lags;
     pvalue = fcdf(fstat, lags, nSamples_regress-(nNodes_rest+2)*lags - 1, 'upper');
else
    fstat = (rss_reduced/rss_full - 1) ...
        * (nSamples_regress - (nNodes_rest+2) * lags)/lags;
    pvalue = fcdf(fstat, lags, nSamples_regress-(nNodes_rest+2)*lags, 'upper');
end
    
    
%%
if plotFlag
    %%
    figure(16)
    h1=histogram(residue_reduced,40,'normalization','pdf');
    h1_bin = h1.BinEdges;
    hold on
    h2=histogram(residue_full,h1_bin,'normalization','pdf');
    hold off
    
    figure(1)
    plot(h1_bin(1:end-1) + h1.BinWidth/2, h2.Values - h1.Values);
    
end


end
