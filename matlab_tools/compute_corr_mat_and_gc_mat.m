function [corr_mat, bvgc_mat, mvgc_mat, bvgc_value, mvgc_value, ...
    bvgc_pvalue_mat, mvgc_pvalue_mat, ...
    bvgc_fstat_mat, mvgc_fstat_mat] = ...
    compute_corr_mat_and_gc_mat (sigma_sim, gcLags, ...
    plotFlag, constantFlag, fstat_effect_set)
    
if nargin < 3
    plotFlag = 0;
end

if nargin < 4
    constantFlag = 1;
end

if nargin < 5
    fstat_effect_set = [0 0];
end


%%
[nNodes, nSamples] = size(sigma_sim);
nSamples_regress = nSamples - gcLags;
%%
corr_mat = corr(sigma_sim')-eye(nNodes);

mvgc_pvalue_mat = zeros(nNodes);
bvgc_pvalue_mat = zeros(nNodes);
mvgc_fstat_mat = zeros(nNodes);
bvgc_fstat_mat = zeros(nNodes);
bvgc_value = zeros(nNodes);
mvgc_value = zeros(nNodes);

%%
for i = 1:nNodes
    for j = 1:nNodes
        if i ~= j 
            %%
            x1 = sigma_sim(i,:);
            x2 = sigma_sim(j,:);
            xrest = sigma_sim(setdiff(1:nNodes, [i, j]) , :);
            [fstat_bi, pvalue_bi, rss_reduced, rss_full] = ...
                measure_gc_bivariate(x1, x2, gcLags, ...
                0, constantFlag, fstat_effect_set(1));
            [fstat_multi, pvalue_multi, rss_reduced_multi, rss_full_multi] = ...
                measure_gc_multivariate(x1, x2, xrest, gcLags, ...
                0, constantFlag, fstat_effect_set(2));

            bvgc_value(i,j) = log(rss_reduced./(nSamples_regress-gcLags-1)./...
                rss_full.*(nSamples_regress - 2*gcLags-1));
            mvgc_value(i,j) = log(rss_reduced_multi./...
                (nSamples_regress-(nNodes-1)*gcLags-1)./...
                rss_full_multi.*(nSamples_regress - nNodes*gcLags - 1));
            
            mvgc_pvalue_mat(i, j) = pvalue_multi; % if p < 0.05, j cause i 
            bvgc_pvalue_mat(i, j) = pvalue_bi;
            mvgc_fstat_mat(i, j) = fstat_multi; % if p < 0.05, j cause i 
            bvgc_fstat_mat(i, j) = fstat_bi;

        end
    end
end

bvgc_pvalue_mat = bvgc_pvalue_mat + eye(nNodes);
% bvgc_mat = bvgc_pvalue_mat<0.05/nNodes/(nNodes-1) ;
bvgc_mat = bvgc_pvalue_mat<0.01/nNodes/(nNodes-1) ;


mvgc_pvalue_mat = mvgc_pvalue_mat + eye(nNodes);
% mvgc_mat = mvgc_pvalue_mat<0.05/nNodes/(nNodes-1) ;
mvgc_mat = mvgc_pvalue_mat<0.01/nNodes/(nNodes-1) ;

%%
if plotFlag
    %%
    subplot(2,3,1)
    imagesc(corr_mat)
    title('Correlation matrix')
    
    subplot(2,3,2)
    imagesc(bvgc_mat)
    title('GC_{bi.} with bon. correction')

    subplot(2,3,3)
    imagesc(mvgc_mat)
    title('GC_{multi.} with bon. correction')
    
    subplot(2,3,4)
    
    x = 0:0.01:20;
    nNodes = 10;
    if constantFlag
        y = fpdf(x, gcLags, nSamples-nNodes*gcLags-1);
    else
        y = fpdf(x, gcLags, nSamples-nNodes*gcLags);
    end
    
    histogram(bvgc_fstat_mat(bvgc_fstat_mat<20),'normalization','pdf')

    hold on
    histogram(mvgc_fstat_mat(mvgc_fstat_mat<20),'normalization','pdf')
    hold on
    plot(x,y,'k','linewidth',1.5);
    hold off
    xlim([0 20])
    
    subplot(2,3,5)
    imagesc(bvgc_fstat_mat)
    title('bivariate gc, fstat')
    
    subplot(2,3,6)
    imagesc(mvgc_fstat_mat)
    title('multi-variate gc, fstat')
end

end
