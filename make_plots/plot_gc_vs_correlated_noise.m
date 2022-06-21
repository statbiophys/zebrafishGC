function gc_var_plus_correlated_noise
% GC is known to be sensitive to correlated noise. Here we visualize the effect. 

nNodes = 10;
maxLags = 2;
params.maxLags = 2;
plotFlag = 1;
tf = 2000;

gRefMat = zeros(nNodes);
links = [1 6; 1 3; 1 8; 6 7; 3 5; 5 8; ...
         8 9; 9 2; 2 10; 5 10; 7 4; 4 10; 6 5; 9 4; 1 4];
for ilink = 1:size(links,1)  
    gRefMat(links(ilink,2), links(ilink,1)) = 1;
end

%%
c_var = 0.3; 
G0 = zeros(nNodes, nNodes, maxLags);
G0(:,1:5,1) = gRefMat(:,1:5) * c_var; 
G0(:,6:10,1) = gRefMat(:,6:10) * (-c_var); 
G0(:,:,2) = G0(:,:,1);

params.G = G0;
params.maxLags = 2;
params.mu = ones(nNodes,1)*(-2);
params.noise_std = 0.01; 

[sigma_sim_var] = simulate_neuron_var(nNodes, tf*2, params);

%%
plotFlag = 1;
[corr_mat_var, bvgc_mat_var, mvgc_mat_var, ...
    bvgc_pvalue_mat_var, mvgc_pvalue_mat_var, ...
    bvgc_fstat_mat_var, mvgc_fstat_mat_var] = ...
    compute_corr_mat_and_gc_mat (sigma_sim_var, maxLags, plotFlag);

%%
figure(2)
imagesc(gRefMat)

%% add correlated noise

xi = randn(1,tf*2)*0.2;


%
figure(3)

[corr_mat_var, bvgc_mat_var, mvgc_mat_var, ...
    bvgc_pvalue_mat_var, mvgc_pvalue_mat_var, ...
    bvgc_fstat_mat_var, mvgc_fstat_mat_var] = ...
    compute_corr_mat_and_gc_mat (sigma_sim_var+repmat(xi,nNodes,1), maxLags, plotFlag);


%%
%
c_glm = 1; 
G0(:,1:5,1) = gRefMat(:,1:5) * c_glm; 
G0(:,6:10,1) = gRefMat(:,6:10) * (-c_glm); 
G0(:,:,2) = G0(:,:,1);
%
params.G = G0;
params.method = 'naive';
dt_sim = 1;
params.dt_sim = dt_sim;
params.tf = tf;
params.lf = floor(tf / dt_sim);
params.spike_subsample = 0;

[sigma_sim_glm] = simulate_neuron_glm(nNodes, tf*2, params); %,1);
tau = 20;
params.tau_ca = tau;
sigma_sim_glm_cont = spike_to_calcium(sigma_sim_glm, params); 

%%
figure(5)
[corr_mat_glm_cont, bvgc_mat_glm_cont, mvgc_mat_glm_cont, ...
    bvgc_pvalue_mat_glm_cont, mvgc_pvalue_mat_glm_cont, ...
    bvgc_fstat_mat_glm_cont, mvgc_fstat_mat_glm_cont] = ...
    compute_corr_mat_and_gc_mat (sigma_sim_glm_cont, ...
    maxLags, plotFlag);

%%

xi = randn(1,tf)*0.2;

figure(6)
[corr_mat_glm_cont, bvgc_mat_glm_cont, mvgc_mat_glm_cont, ...
    bvgc_pvalue_mat_glm_cont, mvgc_pvalue_mat_glm_cont, ...
    bvgc_fstat_mat_glm_cont, mvgc_fstat_mat_glm_cont] = ...
    compute_corr_mat_and_gc_mat (sigma_sim_glm_cont+repmat(xi,nNodes,1), ...
    maxLags, plotFlag);

%% plot error rate vs. correlated noise amplitude

noise_amp2 = 10.^(-2:0.25:2);  
lna = length(noise_amp2);

plotFlag = 0;
nmc = 10;
fpr_bv2 = zeros(nmc, lna);
fnr_bv2 = zeros(nmc, lna);
fpr_mv2 = zeros(nmc, lna);
fnr_mv2 = zeros(nmc, lna);

c_glm = 1; 
G0(:,1:5,1) = gRefMat(:,1:5) * c_glm; 
G0(:,6:10,1) = gRefMat(:,6:10) * (-c_glm); 
G0(:,:,2) = G0(:,:,1);
%
params.G = G0;
params.method = 'naive';
dt_sim = 1;
params.dt_sim = dt_sim;
params.tf = tf;
params.lf = floor(tf / dt_sim);
params.spike_subsample = 0;



for imc = 1:nmc
    %%
    [sigma_sim_glm] = simulate_neuron_glm(nNodes, tf*2, params); 
    tau = 20;
    params.tau_ca = tau;
    sigma_sim_glm_cont = spike_to_calcium(sigma_sim_glm, params); 
    %%

    for inoise = 1:lna
        
        noise_amp = noise_amp2(inoise);
        
        xi = randn(1,tf)*noise_amp;
       
        [~, bvgc_mat_glm_cont, mvgc_mat_glm_cont] = ...
            compute_corr_mat_and_gc_mat (sigma_sim_glm_cont+repmat(xi,nNodes,1), ...
            maxLags, plotFlag);
        
        [fpr_bv, fnr_bv] = compute_error_rate_adjmat(gRefMat, bvgc_mat_glm_cont);
        fpr_bv2(imc, inoise) = fpr_bv;
        fnr_bv2(imc, inoise) = fnr_bv;
    
        [fpr_mv, fnr_mv] = compute_error_rate_adjmat(gRefMat, mvgc_mat_glm_cont);
        fpr_mv2(imc, inoise) = fpr_mv;
        fnr_mv2(imc, inoise) = fnr_mv;
    
    end

end
%%
figure(1)

subplot(1,2,1)
errorbar(noise_amp2, mean(fpr_bv2),std(fpr_bv2),'-o')
hold on
errorbar(noise_amp2, mean(fpr_mv2),std(fpr_mv2),'-o')
hold off
ylabel('false positive rate')
xlabel('noise amplitude')
ylim([0 1])
set(gca,'xscale','log','fontsize',12)

subplot(1,2,2)
errorbar(noise_amp2, mean(fnr_bv2),std(fnr_bv2),'-o')
hold on
errorbar(noise_amp2, mean(fnr_mv2),std(fnr_mv2),'-o')
hold off
l=legend('BVGC','MVGC');
l.Location = 'northwest';
ylabel('false negative rate')
xlabel('noise amplitude')
ylim([0 1])
set(gca,'xscale','log','fontsize',12)

%%
figure(14)

n2 = [1e-1, 1, 1e1];
for ii = 1:3
    subplot(3,1,ii)
    xi = randn(1,tf)*n2(ii);
    ss = sigma_sim_glm_cont+repmat(xi,nNodes,1);
    
    plot(ss(1,:)+std(ss(1,:))*3)
    hold on
    plot(ss(4,:))
    plot(ss(10,:)-std(ss(10,:))*3)
    hold off
    xlim([600 700])
    title(['common noise amplitude = ' num2str(n2(ii))])
%     yticks([])
ylabel('f')
    set(gca,'fontsize',12)
end
xlabel('t (s)')
sgtitle('GLM-Calcium + common noise')




%%
%% plot error rate vs. correlated noise amplitude

noise_amp2 = 10.^(-4:0.25:0);  
lna = length(noise_amp2);

plotFlag = 0;
nmc = 10;
fpr_bv_var2 = zeros(nmc, lna);
fnr_bv_var2 = zeros(nmc, lna);
fpr_mv_var2 = zeros(nmc, lna);
fnr_mv_var2 = zeros(nmc, lna);

c_var = 0.3; 
G0 = zeros(nNodes, nNodes, maxLags);
G0(:,1:5,1) = gRefMat(:,1:5) * c_var; %0.1265;
G0(:,6:10,1) = gRefMat(:,6:10) * (-c_var); %(-0.1265);
G0(:,:,2) = G0(:,:,1);

params.G = G0;
params.maxLags = 2;
params.mu = ones(nNodes,1)*(-2);
params.noise_std = 0.01; %0.01;


for imc = 1:nmc

    
    [sigma_sim_var] = simulate_neuron_var(nNodes, tf*2, params);
    sigma_sim_var = sigma_sim_var(:,tf+1:end);
    
    
  
    for inoise = 1:lna
        noise_amp = noise_amp2(inoise);
        xi = randn(1,tf)*noise_amp;
       
        [~, bvgc_mat_var, mvgc_mat_var] = ...
            compute_corr_mat_and_gc_mat (sigma_sim_var+repmat(xi,nNodes,1), ...
            maxLags, plotFlag);
        
        [fpr_bv, fnr_bv] = compute_error_rate_adjmat(gRefMat, bvgc_mat_var);
        fpr_bv_var2(imc, inoise) = fpr_bv;
        fnr_bv_var2(imc, inoise) = fnr_bv;
    
        [fpr_mv, fnr_mv] = compute_error_rate_adjmat(gRefMat, mvgc_mat_var);
        fpr_mv_var2(imc, inoise) = fpr_mv;
        fnr_mv_var2(imc, inoise) = fnr_mv;
    
    end

end

%%
figure(13)

n2 = [1e-3, 1e-2, 1e-1];
for ii = 1:3
    subplot(3,1,ii)
    xi = randn(1,tf)*n2(ii);
    ss = sigma_sim_var+repmat(xi,nNodes,1);
    
    plot(ss(1,:)+std(ss(1,:))*3)
    hold on
    plot(ss(4,:))
    plot(ss(10,:)-std(ss(10,:))*3)
    hold off
    xlim([300 400])
    title(['common noise amplitude = ' num2str(n2(ii))])
%     yticks([])
    ylabel('f')
    set(gca,'fontsize',12)
end
xlabel('t (s)')
sgtitle('GP + common noise')
%%
figure(3)
noise_amp2 = 10.^(-4:0.25:0);  %[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28];

subplot(1,2,1)
errorbar(noise_amp2, mean(fpr_bv_var2),std(fpr_bv_var2),'-o')
hold on
errorbar(noise_amp2, mean(fpr_mv_var2),std(fpr_mv_var2),'-o')
plot([1e-2 1e-2],[0 1],'color',[0 0 0]+0.65,'linewidth',1.5)
hold off


ylabel('false positive rate')
xlabel('common noise amplitude')
ylim([0 0.2])
set(gca,'xscale','log','fontsize',12)


subplot(1,2,2)
errorbar(noise_amp2, mean(fnr_bv_var2),std(fnr_bv_var2),'-o')
hold on
errorbar(noise_amp2, mean(fnr_mv_var2),std(fnr_mv_var2),'-o')
plot([1e-2 1e-2],[0 1],'color',[0 0 0]+0.65,'linewidth',1.5)
hold off
l=legend('BVGC','MVGC','ind. noise');
l.Location = 'northwest';
ylabel('false negative rate')
xlabel('common noise amplitude')
ylim([0 1])
set(gca,'xscale','log','fontsize',12)



sgtitle('GP + common noise')

end








