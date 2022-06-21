function plot_fig_fnr_fpr_vs_lag
% For synthetic data with true underlying interaction network,
% we ask the question, how does the time lag used in the GC analysis
% change the resulting network.
% This is to plot figure 2.FGH in the manuscript.

%% 1. Initialize the network
nNodes = 10;
maxLags = 3;
params.maxLags = 3;
plotFlag = 1;
tf = 300; %4000;

gRefMat = zeros(nNodes);
links = [1 6; 1 3; 1 8; 6 7; 3 5; 5 8; ...
         8 9; 9 2; 2 10; 5 10; 7 4; 4 10; 6 5; 9 4; 1 4];
for ilink = 1:size(links,1)  
    gRefMat(links(ilink,2), links(ilink,1)) = 1;
end

%%
nmc = 10;
c = 1;
G0 = zeros(nNodes, nNodes, maxLags);
G0(:,1:5,1) = gRefMat(:,1:5) * c; %0.1265;
G0(:,6:10,1) = gRefMat(:,6:10) * (-c); %(-0.1265);
G0(:,:,2) = G0(:,:,1);
G0(:,:,3) = G0(:,:,1);

params.G = G0;
params.maxLags = 3;
params.mu = ones(nNodes,1)*(-2);
params.noise_std = 0.01; %0.01;

params.method = 'naive';
dt_sim = 1;
params.dt_sim = dt_sim;
params.tf = tf;
params.lf = floor(tf / dt_sim);
params.spike_subsample = 0;

nmc = 10;

lag2 = 1:20; 
nl = length(lag2);

    
bvgc_lag_var = zeros(nNodes, nNodes, nl, nmc);
mvgc_lag_var = zeros(nNodes, nNodes, nl, nmc);
bvgc_pvalue_lag_var = zeros(nNodes, nNodes, nl, nmc);
mvgc_pvalue_lag_var = zeros(nNodes, nNodes, nl, nmc);

bvgc_lag_glmca = zeros(nNodes, nNodes, nl, nmc);
mvgc_lag_glmca = zeros(nNodes, nNodes, nl, nmc);
bvgc_pvalue_lag_glmca = zeros(nNodes, nNodes, nl, nmc);
mvgc_pvalue_lag_glmca = zeros(nNodes, nNodes, nl, nmc);


for imc = 1:nmc
    imc = imc

    c_glm = 0.6; 
    G0 = zeros(nNodes, nNodes, maxLags);
    G0(:,1:5,1) = gRefMat(:,1:5) * c_glm; 
    G0(:,6:10,1) = gRefMat(:,6:10) * (-c_glm); 
    G0(:,:,2) = G0(:,:,1);
    G0(:,:,3) = G0(:,:,1);
    params.G = G0;

    [sigma_sim_glm] = simulate_neuron_glm(nNodes, tf, params);
    tau = 20;
    params.tau_ca = tau;
    sigma_sim_glm_cont = spike_to_calcium(sigma_sim_glm, params); 
    
    %%
    c_var = 0.2; 
    G0 = zeros(nNodes, nNodes, maxLags);
    G0(:,1:5,1) = gRefMat(:,1:5) * c_var; 
    G0(:,6:10,1) = gRefMat(:,6:10) * (-c_var); 
    G0(:,:,2) = G0(:,:,1);
    G0(:,:,3) = G0(:,:,1);
    params.G = G0;
    
    [sigma_sim_var] = simulate_neuron_var(nNodes, tf, params);

    %%
    plotFlag = 0;
    constantFlag = 0;
    % centering sigma_var first

    sigma_sim = sigma_sim_glm_cont;
    sigma_sim = sigma_sim - ...
        repmat(mean(sigma_sim,2),1,tf);

    for ilag  = 1:nl
        mylag = lag2(ilag);
        [~,  bvgc, mvgc, ...
            bvgc_value, mvgc_value] = ...
            compute_corr_mat_and_gc_mat (sigma_sim, mylag, plotFlag, constantFlag);
        bvgc_lag_glmca(:, :, ilag, imc) = bvgc;
        mvgc_lag_glmca(:, :, ilag, imc) = mvgc;
        bvgc_pvalue_lag_glmca(:,:, ilag, imc) = bvgc_value;
        mvgc_pvalue_lag_glmca(:,:, ilag, imc) = mvgc_value;
    end
    
    %%
    sigma_sim = sigma_sim_var;

    sigma_sim = sigma_sim - ...
        repmat(mean(sigma_sim,2),1,tf);

    for ilag  = 1:nl
        mylag = lag2(ilag);
        [~, bvgc, mvgc, ...
            bvgc_value, mvgc_value] = ...
            compute_corr_mat_and_gc_mat ...
            (sigma_sim, mylag, plotFlag, constantFlag);
        bvgc_lag_var(:,:, ilag, imc) = bvgc;
        mvgc_lag_var(:,:, ilag, imc) = mvgc;
        bvgc_pvalue_lag_var(:,:, ilag, imc) = bvgc_value;
        mvgc_pvalue_lag_var(:,:, ilag, imc) = mvgc_value;
    end
end

%% Compute fpr fnr

fpr2_bv_var = zeros(nl, nmc);
fnr2_bv_var = zeros(nl, nmc);
fpr2_mv_var = zeros(nl, nmc);
fnr2_mv_var = zeros(nl, nmc);

fpr2_bv_glmca = zeros(nl, nmc);
fnr2_bv_glmca = zeros(nl, nmc);
fpr2_mv_glmca = zeros(nl, nmc);
fnr2_mv_glmca = zeros(nl, nmc);

for il = 1:nl
    for imc = 1:nmc
        %%
        [fpr, fnr] = ...
            compute_error_rate_adjmat(gRefMat, ...
            bvgc_lag_glmca(:,:,il,imc));
        fpr2_bv_glmca(il, imc) = fpr;
        fnr2_bv_glmca(il, imc) = fnr;
        
        %%
        [fpr, fnr] = ...
            compute_error_rate_adjmat(gRefMat, ...
            mvgc_lag_glmca(:,:,il,imc));
        fpr2_mv_glmca(il, imc) = fpr;
        fnr2_mv_glmca(il, imc) = fnr;
        
        %%
        [fpr, fnr] = ...
            compute_error_rate_adjmat(gRefMat, ...
            bvgc_lag_var(:,:,il,imc));
        fpr2_bv_var(il, imc) = fpr;
        fnr2_bv_var(il, imc) = fnr;
        
        %%
        [fpr, fnr] = ...
            compute_error_rate_adjmat(gRefMat, ...
            mvgc_lag_var(:,:,il,imc));
        fpr2_mv_var(il, imc) = fpr;
        fnr2_mv_var(il, imc) = fnr;
    end
end


%%
mean_prc_sig_bvgc = squeeze(mean(sum(sum(bvgc_lag_var,1),2),4))/(nNodes)/(nNodes-1);

std_prc_sig_bvgc = squeeze(std(sum(sum(bvgc_lag_var,1),2),0,4))/(nNodes)/(nNodes-1);

mean_prc_sig_mvgc = squeeze(mean(sum(sum(mvgc_lag_var,1),2),4))/(nNodes)/(nNodes-1);

std_prc_sig_mvgc = squeeze(std(sum(sum(mvgc_lag_var,1),2),0,4))/(nNodes)/(nNodes-1);

mean_avg_sig_gcvalue_bvgc = squeeze(nanmean(sum(sum(bvgc_pvalue_lag_var.*bvgc_lag_var,1),2)./...
    sum(sum(bvgc_lag_var,1),2),4)); 
std_avg_sig_gcvalue_bvgc = squeeze(nanstd(sum(sum(bvgc_pvalue_lag_var.*bvgc_lag_var,1),2)./...
    sum(sum(bvgc_lag_var,1),2),0,4));
mean_avg_sig_gcvalue_mvgc = squeeze(nanmean(sum(sum(mvgc_pvalue_lag_var.*mvgc_lag_var,1),2)./...
    sum(sum(mvgc_lag_var,1),2),4));
std_avg_sig_gcvalue_mvgc = squeeze(nanstd(sum(sum(mvgc_pvalue_lag_var.*mvgc_lag_var,1),2)./...
    sum(sum(mvgc_lag_var,1),2),0,4));


mean_avg_gcvalue_bvgc = squeeze(mean(sum(sum(bvgc_pvalue_lag_var,1),2),4))/(nNodes)/(nNodes-1);
std_avg_gcvalue_bvgc = squeeze(std(sum(sum(bvgc_pvalue_lag_var,1),2),0,4))/(nNodes)/(nNodes-1);

mean_avg_gcvalue_mvgc = squeeze(mean(sum(sum(mvgc_pvalue_lag_var,1),2),4))/(nNodes)/(nNodes-1);
std_avg_gcvalue_mvgc = squeeze(std(sum(sum(mvgc_pvalue_lag_var,1),2),0,4))/(nNodes)/(nNodes-1);

mean_fpr_bvgc = mean(fpr2_bv_var,2);
std_fpr_bvgc = std(fpr2_bv_var,0,2);
mean_fpr_mvgc = mean(fpr2_mv_var,2);
std_fpr_mvgc = std(fpr2_mv_var,0,2);

mean_fnr_bvgc = mean(fnr2_bv_var,2);
std_fnr_bvgc = std(fnr2_bv_var,0,2);
mean_fnr_mvgc = mean(fnr2_mv_var,2);
std_fnr_mvgc = std(fnr2_mv_var,0,2);


%% 2. Plot for VAR

python_green = [80, 161, 37]/256;
python_purple = [115, 26, 162]/256;

figure(1)
subplot(3,2,1)
errorbar(lag2, squeeze(mean(sum(sum(bvgc_lag_var,1),2),4))/(nNodes)/(nNodes-1), ...
    squeeze(std(sum(sum(bvgc_lag_var,1),2),0,4))/(nNodes)/(nNodes-1),'-o',...
    'color', python_purple)
hold on
errorbar(lag2, squeeze(mean(sum(sum(mvgc_lag_var,1),2),4))/(nNodes)/(nNodes-1),...
    squeeze(std(sum(sum(mvgc_lag_var,1),2),0,4))/(nNodes)/(nNodes-1),'-x',...
    'color', python_green )
plot([0 20],[1 1]*sum(sum(gRefMat))./(nNodes*(nNodes-1)),'-',...
    'color',[0 0 0]+0.65)
plot([3 3],[0 1],'-','color',[0 0 0]+0.65)
hold off
xlabel('Lag')
ylabel('% significant links')
title('VAR')
l=legend('BVGC','MVGC','ground truth');
l.Location = 'northeast';
xlim([0 20])
ylim([0 0.32])
set(gca,'fontsize',12)


subplot(3,2,2)
errorbar(lag2, ...
    squeeze(nanmean(sum(sum(bvgc_pvalue_lag_var.*bvgc_lag_var,1),2)./...
    sum(sum(bvgc_lag_var,1),2),4)), ...
    squeeze(nanstd(sum(sum(bvgc_pvalue_lag_var.*bvgc_lag_var,1),2)./...
    sum(sum(bvgc_lag_var,1),2),0,4)),'-o', ...
    'color', python_purple);

hold on
errorbar(lag2, ...
    squeeze(nanmean(sum(sum(mvgc_pvalue_lag_var.*mvgc_lag_var,1),2)./...
    sum(sum(mvgc_lag_var,1),2),4)), ...
    squeeze(nanstd(sum(sum(mvgc_pvalue_lag_var.*mvgc_lag_var,1),2)./...
    sum(sum(mvgc_lag_var,1),2),0,4)),'-x',...
    'color', python_green);
plot([3 3],[0 1],'-','color',[0 0 0]+0.65)

hold off
ylim([0.05 0.25])
title('VAR')
xlabel('Lag')
ylabel('Average sig. GC value')
set(gca,'fontsize',12)
% set(gca,'yscale','log')

subplot(3,2,3)
errorbar(lag2, squeeze(mean(sum(sum(bvgc_pvalue_lag_var,1),2),4))/(nNodes)/(nNodes-1), ...
    squeeze(std(sum(sum(bvgc_pvalue_lag_var,1),2),0,4))/(nNodes)/(nNodes-1),...
    '-o', 'color', python_purple)
hold on
errorbar(lag2, squeeze(mean(sum(sum(mvgc_pvalue_lag_var,1),2),4))/(nNodes)/(nNodes-1),...
    squeeze(std(sum(sum(mvgc_pvalue_lag_var,1),2),0,4))/(nNodes)/(nNodes-1),...
    '-x', 'color', python_green)
plot([3 3],[0 1],'-','color',[0 0 0]+0.65)

hold off
xlabel('Lag')
ylabel('Average GC value')
title('VAR')
set(gca,'fontsize',12)
ylim([0 0.05])


% compute fpr fnr
%
subplot(3,2,4)
errorbar(lag2, mean(fpr2_bv_var,2), std(fpr2_bv_var,0,2), '-o', ...
    'color', python_purple);
hold on
errorbar(lag2, mean(fpr2_mv_var,2), std(fpr2_mv_var,0,2), '-x', ...
    'color', python_green);
plot([3 3],[0 1],'-','color',[0 0 0]+0.65)

hold off
xlabel('Lag')
ylabel('False postive rate')
title('VAR')
% l=legend('BVGC','MVGC');
% l.Location = 'northwest';
set(gca,'fontsize',12)
% xlim([0 15])
ylim([0, 0.05])

%
subplot(3,2,5)
errorbar(lag2, mean(fnr2_bv_var,2), std(fnr2_bv_var,0,2), '-o', ...
    'color', python_purple);
hold on
errorbar(lag2, mean(fnr2_mv_var,2), std(fnr2_mv_var,0,2), '-x', ...
    'color', python_green);
plot([3 3],[0 1],'-','color',[0 0 0]+0.65)

hold off
xlabel('Lag')
ylabel('False negative rate')
title('VAR')
% l=legend('BVGC','MVGC');
% l.Location = 'northwest';
set(gca,'fontsize',12)

end



