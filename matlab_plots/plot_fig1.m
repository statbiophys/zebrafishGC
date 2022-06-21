function plot_fig1
% function plot_fig1 
%
% Simulate toy dynamics (VAR, GLM, and GLM+calcium) on a toy network of 10
% neurons. Compute correlation, and granger causality (GC). Analyze the GC
% performance as a function of connectivity strength of the network.
%
% This code generate fig1 in the manuscript Chen, Ginoux, Wyart, Mora,
% Walczak (2022).


matlab_red = [0.8500 0.3250 0.0980] ;
matlab_yellow = [0.9290 0.6940 0.1250];

%% 1.1 Initialize the network

nNodes = 10;
maxLags = 2;
params.maxLags = 2;
plotFlag = 1;
tf = 4000;

% hand-picked gRefMat, the underlying true connectivity matrix
gRefMat = zeros(nNodes);
links = [1 6; 1 3; 1 8; 6 7; 3 5; 5 8; ...
         8 9; 9 2; 2 10; 5 10; 7 4; 4 10; 6 5; 9 4; 1 4];
for ilink = 1:size(links,1)  
    gRefMat(links(ilink,2), links(ilink,1)) = 1;
end

%% 1.2 Plot the true underlying network
figure(1)
imagesc(gRefMat')
ylabel('from neuron #')
xlabel('to neuron #')
axis square
set(gca, 'fontsize', 12)
xticks(1:10)
yticks(1:10)
colormap([1 1 1; 0 0 0])


%% 2.1 Simulate dynamics: VAR
% 2.1.1 Initialize the system
c_var = 0.3; 
G0 = zeros(nNodes, nNodes, maxLags);
G0(:,1:5,1) = gRefMat(:,1:5) * c_var;  % excitatory neurons
G0(:,6:10,1) = gRefMat(:,6:10) * (-c_var); % inhibitory neurons
G0(:,:,2) = G0(:,:,1);

params.G = G0;
params.maxLags = 2;
params.mu = ones(nNodes,1)*(-2);
params.noise_std = 0.01;

% 2.1.2 Run the VAR dynamics
[sigma_sim_var] = simulate_neuron_var(nNodes, tf, params);

% 2.1.2 Visualize the trajectory
figure(2)
plot(sigma_sim_var(1,:)+0.06)
hold on
plot(sigma_sim_var(4,:))
plot(sigma_sim_var(10,:)-0.06)
hold off
xlim([200 300])
xlabel('Time')

%% 2.2 Simulate dynamics: GLM and GLM-calcium
% 2.2.1 Initialize 
c_glm = 1.5; 
G0(:,1:5,1) = gRefMat(:,1:5) * c_glm; 
G0(:,6:10,1) = gRefMat(:,6:10) * (-c_glm); 
G0(:,:,2) = G0(:,:,1);

params.G = G0;
params.method = 'naive';
dt_sim = 1;
params.dt_sim = dt_sim;
params.tf = tf;
params.lf = floor(tf / dt_sim);
params.spike_subsample = 0; % no subsampling. 

% 2.2.2 Simulate GLM

[sigma_sim_glm] = simulate_neuron_glm(nNodes, tf, params);
        
% 2.2.3 Convolute with exponential kernel to simulate GLM-calcium

tau = 20;
params.tau_ca = tau;
sigma_sim_glm_cont = spike_to_calcium(sigma_sim_glm, params); 


% 2.2.4 Visualize synthetic trajectories: GLM
figure(3)
subplot(3,1,1)
stem(sigma_sim_glm(1,:))
xlim([200 300])

subplot(3,1,2)
stem(sigma_sim_glm(4,:), 'color', matlab_red)
xlim([200 300])

subplot(3,1,3)
stem(sigma_sim_glm(10,:), 'color',matlab_yellow)
xlim([200 300])

% 2.2.5 Visualize synthetic trajectories: GLM-calcium
figure(4)
plot(sigma_sim_glm_cont(1,:)+10)
hold on
plot(sigma_sim_glm_cont(4,:)+5)
plot(sigma_sim_glm_cont(10,:)-4)
hold off
xlim([200 300])

%% 3. Compute pairwise correlation and Granger Causality among all pairs
plotFlag = 0; constantFlag = 1;
[corr_mat_var, bvgc_mat_var, mvgc_mat_var, ...
    bvgc_pvalue_mat_var, mvgc_pvalue_mat_var, ...
    bvgc_fstat_mat_var, mvgc_fstat_mat_var] = ...
    compute_corr_mat_and_gc_mat ...
    (sigma_sim_var - repmat(mean(sigma_sim_var,2),1,tf), maxLags, plotFlag, constantFlag);

[corr_mat_glm, bvgc_mat_glm, mvgc_mat_glm, ...
    bvgc_pvalue_mat_glm, mvgc_pvalue_mat_glm, ...
    bvgc_fstat_mat_glm, mvgc_fstat_mat_glm] = ...
    compute_corr_mat_and_gc_mat (...
    sigma_sim_glm - repmat(mean(sigma_sim_glm,2),1,tf), ...
    maxLags, plotFlag, constantFlag);

[corr_mat_glm_cont, bvgc_mat_glm_cont, mvgc_mat_glm_cont, ...
    bvgc_pvalue_mat_glm_cont, mvgc_pvalue_mat_glm_cont, ...
    bvgc_fstat_mat_glm_cont, mvgc_fstat_mat_glm_cont] = ...
    compute_corr_mat_and_gc_mat ...
    (sigma_sim_glm_cont - repmat(mean(sigma_sim_glm_cont,2),1,tf), ...
    maxLags, plotFlag, constantFlag);

%% 3.1 Visualize the correlation matrices for the three example trajectories
figure(11)
subplot(3,1,1)
imagesc(corr_mat_var)
title('VAR')

subplot(3,1,2)
imagesc(corr_mat_glm)
title('GLM')

subplot(3,1,3)
imagesc(corr_mat_glm_cont)
title('GLM-calcium')


for i = 1:3
    subplot(3,1,i)
    c=colorbar()
%     caxis([-0.4 0.4])
    axis square
    xlabel('to neuron')
    ylabel('from neuron')
    xticks([1 10])
    yticks([1 10])
    set(gca,'fontsize',12)
    ylabel(c,'correlation')
end

%% 3.2 Visualize the GC matrices for the three example trajectories

figure(12)
subplot(3,2,1)
imagesc(bvgc_mat_var')
hold on
plot(links(:,2), links(:,1),'r.','MarkerSize',12)
hold off
axis square
title('VAR, BVGC')

subplot(3,2,2)
imagesc(mvgc_mat_var')
colormap([0.98 0.98 0.92; 0 0 0])
overlay_refmat(links)
axis square
title('VAR, MVGC')

subplot(3,2,3)
imagesc(bvgc_mat_glm')
overlay_refmat(links)
axis square
title('GLM, BVGC')

subplot(3,2,4)
imagesc(mvgc_mat_glm')
colormap([0.98 0.98 0.92; 0 0 0])
overlay_refmat(links)
axis square
title('GLM, MVGC')

subplot(3,2,5)
imagesc(bvgc_mat_glm_cont')
overlay_refmat(links)
axis square
title('GLM-calcium, BVGC')

subplot(3,2,6)
imagesc(mvgc_mat_glm_cont')
overlay_refmat(links)
axis square
title('GLM-calcium, MVGC')

for i = 1:6
    subplot(3,2,i)
    axis square
    xlabel('to neuron')
    ylabel('from neuron')
    xticks([1 10])
    yticks([1 10])
    set(gca,'fontsize',12)
end

%% 4. Vary the connectivity strength c, run dynamics, apply GC analysis
%     and plot the false positive rate (fpr), false negative rate (fnr) as
%     a function of c (Fig1, panel I,J)

%  4.1 VAR
%  4.1.1 scan through connection strength for VAR. 
cList_var = 0.05:0.05:0.7;
lc = length(cList_var);
tfList_var = 2000; %[500 1000 2000 4000 8000]; % 8000];
ltf = length(tfList_var);
nmc = 10;
params.nmc = nmc;
fpr2_bv_var = zeros(lc, ltf, nmc);
fnr2_bv_var = fpr2_bv_var;
fpr2_mv_var = fpr2_bv_var;
fnr2_mv_var = fpr2_bv_var;

params.mu = ones(nNodes,1)*(-2);

gii1 = -0.1; % diagonal terms of the G matrix, to keep the dynamics stable
for itf = 1:length(tfList_var)
    tf = tfList_var(itf)
    params.tf = tf;
    [fpr2_bv, fnr2_bv, fpr2_mv, fnr2_mv] = ...
        compute_error_rate_adjmat_list(gRefMat, cList_var, params, 'var', gii1);
    fpr2_bv_var(:,itf,:) = fpr2_bv;
    fnr2_bv_var(:,itf,:) = fnr2_bv;
    fpr2_mv_var(:,itf,:) = fpr2_mv;
    fnr2_mv_var(:,itf,:) = fnr2_mv;
end

% 4.1.2. Plot the fpr and fnr of the BVGC and MVGC matrix, vs c
for itf = 1:ltf
    figure(5)
    errorbar(cList_var, mean(fpr2_bv_var(:,itf,:),3), ...
        std(fpr2_bv_var(:,itf,:),0,3)./sqrt(nmc),'o')
    hold on
    errorbar(cList_var, mean(fpr2_mv_var(:,itf,:),3), ...
        std(fpr2_mv_var(:,itf,:),0,3)./sqrt(nmc),'o')
    errorbar(cList_var, mean(fnr2_bv_var(:,itf,:),3), ...
        std(fnr2_bv_var(:,itf,:),0,3)./sqrt(nmc),'o')
    errorbar(cList_var, mean(fnr2_mv_var(:,itf,:),3), ...
        std(fnr2_mv_var(:,itf,:),0,3)./sqrt(nmc),'o')
    hold off
    legend('FPR, bi-variate','FPR, multi-variate',...
        'FNR, bi-variate','FNR, multi-variate')
    pause
end


%  4.2 GLM
%  4.2.1 scan through connection strength for GLM. 
cList = 0.1:0.1:2 ;
lc = length(cList);
tfList = 2000; %[500 1000 2000 4000 8000]; % 8000];
ltf = length(tfList);
nmc = 10;
params.nmc = nmc;
fpr2_bv_glm = zeros(lc, ltf, nmc);
fnr2_bv_glm = fpr2_bv_glm;
fpr2_mv_glm = fpr2_bv_glm;
fnr2_mv_glm = fpr2_bv_glm;

params.mu = ones(nNodes,1)*(-2);

for itf = 1:length(tfList)
    tf = tfList(itf)
    params.tf = tf;
    [fpr2_bv, fnr2_bv, fpr2_mv, fnr2_mv] = ...
        compute_error_rate_adjmat_list(gRefMat, cList, params, 'glm');
    fpr2_bv_glm(:,itf,:) = fpr2_bv;
    fnr2_bv_glm(:,itf,:) = fnr2_bv;
    fpr2_mv_glm(:,itf,:) = fpr2_mv;
    fnr2_mv_glm(:,itf,:) = fnr2_mv;
end

for itf = 1:ltf
    figure()
    errorbar(cList, mean(fpr2_bv_glm(:,itf,:),3), ...
        std(fpr2_bv_glm(:,itf,:),0,3)./sqrt(nmc),'o')
    hold on
    errorbar(cList, mean(fpr2_mv_glm(:,itf,:),3), ...
        std(fpr2_mv_glm(:,itf,:),0,3)./sqrt(nmc),'o')
    errorbar(cList, mean(fnr2_bv_glm(:,itf,:),3), ...
        std(fnr2_bv_glm(:,itf,:),0,3)./sqrt(nmc),'o')
    errorbar(cList, mean(fnr2_mv_glm(:,itf,:),3), ...
        std(fnr2_mv_glm(:,itf,:),0,3)./sqrt(nmc),'o')
    hold off
    legend('FPR, bi-variate','FPR, multi-variate',...
        'FNR, bi-variate','FNR, multi-variate')
end

%% 4.3 GLM-calcium
fpr2_bv_glm_cont = zeros(lc, ltf, nmc);
fnr2_bv_glm_cont = fpr2_bv_glm_cont;
fpr2_mv_glm_cont = fpr2_bv_glm_cont;
fnr2_mv_glm_cont = fpr2_bv_glm_cont;

tau = 20;

for itf = 1:length(tfList)
    tf = tfList(itf)
    params.tf = tf;
%     params.tau_ca = tau;
    [fpr2_bv, fnr2_bv, fpr2_mv, fnr2_mv] = ...
        compute_error_rate_adjmat_list(gRefMat, cList, params, 'glm_calcium');
    fpr2_bv_glm_cont(:,itf,:) = fpr2_bv;
    fnr2_bv_glm_cont(:,itf,:) = fnr2_bv;
    fpr2_mv_glm_cont(:,itf,:) = fpr2_mv;
    fnr2_mv_glm_cont(:,itf,:) = fnr2_mv;
end


%%
for itf = 1:ltf
    figure()
    errorbar(cList, mean(fpr2_bv_glm_cont(:,itf,:),3), ...
        std(fpr2_bv_glm_cont(:,itf,:),0,3)./sqrt(nmc),'o')
    hold on
    errorbar(cList, mean(fpr2_mv_glm_cont(:,itf,:),3), ...
        std(fpr2_mv_glm_cont(:,itf,:),0,3)./sqrt(nmc),'o')
    errorbar(cList, mean(fnr2_bv_glm_cont(:,itf,:),3), ...
        std(fnr2_bv_glm_cont(:,itf,:),0,3)./sqrt(nmc),'o')
    errorbar(cList, mean(fnr2_mv_glm_cont(:,itf,:),3), ...
        std(fnr2_mv_glm_cont(:,itf,:),0,3)./sqrt(nmc),'o')
    hold off
    legend('FPR, bi-variate','FPR, multi-variate',...
        'FNR, bi-variate','FNR, multi-variate')
end
     
end

% Run the dynamics, perform GC, then output the fpr and fnr compared to the groundtruth
function [fpr2_bv, fnr2_bv, fpr2_mv, fnr2_mv] = ...
    compute_error_rate_adjmat_list(gRefMat, cList, params, dynName, gii1)

if nargin < 5
    gii1 = 0;
end

lc = length(cList);
nmc = params.nmc;
fpr2_bv = nan(lc, nmc);
fnr2_bv = nan(lc, nmc);
fpr2_mv = nan(lc, nmc);
fnr2_mv = nan(lc, nmc);
% params.noise_std = 0.01;

nNodes = size(gRefMat,1);
maxLags = params.maxLags;
tf = params.tf;
gcLags = maxLags;

for ic = 1:lc
    c = cList(ic)
    
    G0 = zeros(nNodes, nNodes, maxLags);
    G0(:,1:5,1) = gRefMat(:,1:5) * c; 
    G0(:,6:10,1) = gRefMat(:,6:10) * (-c); 
    G0(:,:,2) = G0(:,:,1);
    for ii = 1:nNodes
        G0(ii,ii,1) = gii1; 
    end

    %
    params.G = G0;
    
    for imc = 1:nmc
        
        switch dynName
            case 'var'
                sigma_sim = simulate_neuron_var(nNodes, tf*2, params);
            case 'glm'
                sigma_sim = simulate_neuron_glm(nNodes, tf*2, params);
            case 'glm_calcium'
                sigma_sim = simulate_neuron_glm(nNodes, tf*2, params);
                tau = params.tau_ca;
                sigma_sim = spike_to_calcium(sigma_sim, params);
%                 sigma_sim = spike_to_calcium(sigma_sim, tau); 
           otherwise
                warning('Unexpected dynamics type. No plot created.')
        end


        if max(abs(sigma_sim(:))) < 1e3

            [~, bvgc_mat, mvgc_mat] = ...
                compute_corr_mat_and_gc_mat (sigma_sim, gcLags, 0);

            [fpr_bv, fnr_bv] = compute_error_rate_adjmat(gRefMat, bvgc_mat);
            fpr2_bv(ic, imc) = fpr_bv;
            fnr2_bv(ic, imc) = fnr_bv;

            [fpr_mv, fnr_mv] = compute_error_rate_adjmat(gRefMat, mvgc_mat);
            fpr2_mv(ic, imc) = fpr_mv;
            fnr2_mv(ic, imc) = fnr_mv;

        end
        
    end
end


end

% overlay the ground truth (red dots) to the plot of GC matrices
function overlay_refmat(links)

hold on
plot(links(:,2), links(:,1),'r.','MarkerSize',12)
hold off

end
