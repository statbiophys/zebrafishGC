function plot_fig6
% Generate adaptive threshold for significance test in granger causality
% analysis, using zebrafish embryonic motoneuron data.
%
% This code generate Fig 6 in the manuscript.

%%
fish_trace_pair = [1 1; 1 2; 3 1; 3 2; 4 1; ...
                   5 1; 5 2; 5 3; 6 1; 6 2];

%%
plotFlag = 0;
nmc = 100; %20
nNodes = 20;
nfish_trace = 10;
bvgc_value = zeros(nNodes, nNodes, nfish_trace);
mvgc_value = zeros(nNodes, nNodes, nfish_trace);
bvgc_fstat = zeros(nNodes, nNodes, nfish_trace);
mvgc_fstat = zeros(nNodes, nNodes, nfish_trace);

cbvgc_value = zeros(nNodes, nNodes, nfish_trace, nmc);
cmvgc_value = zeros(nNodes, nNodes, nfish_trace, nmc);
cbvgc_fstat = zeros(nNodes, nNodes, nfish_trace, nmc);
cmvgc_fstat = zeros(nNodes, nNodes, nfish_trace, nmc);


%%
for ifish_trace = 1:nfish_trace

    %%
    ifish_trace

    
    fish_id = fish_trace_pair(ifish_trace, 1);
    trace_id = fish_trace_pair(ifish_trace, 2);

    fn = ['f' num2str(fish_id) 't' num2str(trace_id) '_macorrected_clean.mat'];
    load(fn, 'dff','f_smooth','disc_f');
    f = f_smooth(:, 1:end-1);
%%
    [nNodes, nSamples] = size(f);
    f = f - repmat(mean(f,2),1,nSamples);
    
    constantFlag = 0;
    for ii = 1:nNodes
        for jj = 1:nNodes
            if ii ~= jj
                x1 = f(ii,:); x2 = f(jj,:);
                xrest = f(setdiff(1:nNodes, [ii jj]) , :);

                lags = 3;    
                [fstat_bi, pvalue_bi, rss_reduced, rss_full] = ...
                    measure_gc_bivariate(x1, x2, lags, plotFlag, constantFlag);

                [fstat_multi, pvalue_multi, rss_reduced_multi, rss_full_multi] = ...
                    measure_gc_multivariate(x1, x2, xrest, lags, ...
                    plotFlag, constantFlag);

                bvgc_value(ii,jj,ifish_trace) = ...
                    log(rss_reduced./(nSamples-lags)./...
                    rss_full.*(nSamples - 2*lags));
                mvgc_value(ii,jj,ifish_trace) = ...
                    log(rss_reduced_multi./(nSamples-(nNodes-1)*lags)./...
                    rss_full_multi.*(nSamples - nNodes*lags));
            

                bvgc_fstat(ii, jj, ifish_trace) = fstat_bi;
                mvgc_fstat(ii, jj, ifish_trace) = fstat_multi;

                kkfish_trace = setdiff(2:nfish_trace, ifish_trace);

                for imc = 1:nmc
                    %%
                    cx2 = random_cyclic_shuffle(x2);
                    [cfstat_bi, cpvalue_bi, crss_reduced, crss_full] = ...
                        measure_gc_bivariate(x1, cx2, lags, plotFlag, constantFlag);

                    [cfstat_multi, cpvalue_multi, crss_reduced_multi, crss_full_multi] = ...
                        measure_gc_multivariate(x1, cx2, xrest, lags, plotFlag, constantFlag);

                    cbvgc_value(ii,jj,ifish_trace, imc) = ...
                        log(crss_reduced./(nSamples-lags)./...
                        crss_full.*(nSamples - 2*lags));
                    cmvgc_value(ii,jj,ifish_trace, imc) = ...
                        log(crss_reduced_multi./(nSamples-(nNodes-1)*lags)./...
                        crss_full_multi.*(nSamples - nNodes*lags));

                    cbvgc_fstat(ii, jj, ifish_trace, imc) = cfstat_bi;
                    cmvgc_fstat(ii, jj, ifish_trace, imc) = cfstat_multi;
                end

            end
        end
    end
end


%% Find the optimal fit for the f-stat distribution for shuffled data
for ifish_trace = 1:10
    ifish_trace
    a = cbvgc_fstat(:,:,ifish_trace,:);
    a = a(:);
    a = a(a~=0);
    ll2 = [];
    for fparam = 1:10
        ll = 0;
        for ii = 1:length(a)
            ll = ll + log(fpdf(a(ii),lags,fparam));
        end
        
        ll = ll/length(a);
        ll2 = [ll2 ll];
    end
    [~, my_argmax] = max(ll2);
        disp(['fparam = ' num2str(fparam) ', maxll at = ' num2str(my_argmax)]);
end

%% BVGC thres for f3t2
figure(11)
x = 0:0.01:100;
yfit0 = fpdf(x,lags,nSamples-2*lags-1);
yfit = fpdf(x,lags,7);

for ifish_trace = 4

    a = bvgc_fstat(:,:,ifish_trace);
    [hv, hb] = histcounts(a(a~=0),100,'normalization','pdf');
    plot((hb(1:end-1)+hb(2:end))/2, hv,'o')
    hold on

    a = cbvgc_fstat(:,:,ifish_trace,:);
    [hv, hb] = histcounts(a(a~=0),1:1:100,'normalization','pdf');
    plot((hb(1:end-1)+hb(2:end))/2, hv,'gx')
    hold on
    plot(x,yfit0,'-','color',[0 0 0]+0.65)
    plot(x,yfit,'k-')

    hold off
    ylim([1e-5 1])
    xlim([1e-2 1e3])

    set(gca,'xscale','log')
    set(gca,'yscale','log')
end

hold on
plot([1 1]*finv(1-0.01/10/11*2,lags,nSamples-2*lags-1),...
    [1e-5,1],'--','color',[0 0 0]+0.65)
plot([1 1]*finv(1-0.01/10/11*2,lags,7),...
    [1e-5,1],'k--')
hold off

xlabel('f stat')
ylabel('probability density')
title('BVGC, motoneuron, f smooth, f3t2')
l=legend('data','shuffled d.','F_{naive}','F_{fit} = F(3,7)');
l.Location='southwest';
set(gca,'fontsize',12)

%%
%%%%%% MVGC thres for f3t2
figure(12)
x = 0:0.01:100;
yfit0 = fpdf(x,lags,nSamples-nNodes*lags-1);
yfit = fpdf(x,lags,10);


for ifish_trace = 4
    a = mvgc_fstat(:,:,ifish_trace);
    [hv, hb] = histcounts(a(a~=0),30,'normalization','pdf');
    plot((hb(1:end-1)+hb(2:end))/2, hv,'o')
    hold on

    a = cmvgc_fstat(:,:,ifish_trace,:);
    %%
    [hv, hb] = histcounts(a(a~=0),1:100,'normalization','pdf');
    plot((hb(1:end-1)+hb(2:end))/2, hv,'gx')
    hold on
    plot(x,yfit0,'-','color',[0 0 0]+0.65)
    plot(x,yfit,'k-')

    hold off
    ylim([1e-5 1])
    xlim([1e-2 1e3])

    set(gca,'xscale','log')
    set(gca,'yscale','log')
end

hold on
plot([1 1]*finv(1-0.01/10/11*2,lags,nSamples-2*lags-1),...
    [1e-5,1],'--','color',[0 0 0]+0.65)
plot([1 1]*finv(1-0.01/10/11*2,lags,10),...
    [1e-5,1],'k--')
hold off

xlabel('f stat')
ylabel('probability density')
title('MVGC, motoneuron, f smooth, f3t2')
l=legend('data','shuffled d.','F_{naive}','F_{fit} = F(3,10)');
l.Location='southwest';
set(gca,'fontsize',12)


end


function [output, index_new, rset] = random_cyclic_shuffle(input, given_index)

[nNodes, nSamples] = size(input);

rset = [];

if nargin < 2
    output = input;
    index_old = repmat(1:nSamples,nNodes,1);
    index_new = index_old;
    
    for i = 1:nNodes
        r = randi(nSamples);
        rset = [rset r];
        output(i,r+1:end) = input(i,1:end-r);
        output(i,1:r) = input(i,end-r+1:end);

        index_new(i,r+1:end) = index_old(i,1:end-r);
        index_new(i,1:r) = index_old(i,end-r+1:end);
    end
else
    for i = 1:nNodes
        output(i,:) = input(i,given_index(i,:));
    end
    index_new = given_index;
end

end

