function simulate_motoneuron_fast_info_prop
% XC 2022-04-22
% Set the information propagation time to be smaller than the sampling rate
%
% Use smaller delta t
nNodes = 10;

sampling_freq = 4 ; % 4 hz
% params.dt_sim = 1/sampling_freq;

dt_sim = 0.0125; %0.025; %0.125;
dt_info = 0.25;
dn_info = floor(dt_info / dt_sim);
params.dt_sim = dt_sim;

t_fire_min = 8 / sampling_freq / dt_sim; % 8 time points continuously
t_fire_max = 16 / sampling_freq / dt_sim ;
t_quiet_min = 20 / sampling_freq / dt_sim ; %30; 
t_quiet_max = 50 / sampling_freq / dt_sim ;

spike_rate =  32; % 0.4;

dt_min = 2 / sampling_freq / dt_sim ; 
dt_max = 10 / sampling_freq / dt_sim ; 

maxLags = 2;


%
tf = 1000 / sampling_freq / dt_sim;
t_burnin = 1000 / sampling_freq / dt_sim;
tf = tf + t_burnin;


%%
% tau2 is in unit of seconds. the transform happens in spike_in_calcium
% and params.dt_sim need to be set to the correct value
tau2 = (0.5:0.5:10); %0.5:0.5:5; %1:20; %[1:5:21]; %[1,2,4,8,16,32,64];
ltau = length(tau2);
nmc = 10;
wipsi_sim = zeros(ltau, 4, nmc);
whtt_sim = zeros(ltau, 4, nmc);
nl = 5; nr = 5;

constantFlag = 1;
plotFlag = 0;



%%
for imc = 1:nmc
    %%
    imc = imc


    mydrive_left = zeros(1, ceil(tf + dn_info * nNodes/2));
    mydrive_right = zeros(1, ceil(tf + dn_info * nNodes/2));
    
    t = 1;
    while t < tf
        dt_quite = randi(t_quiet_max - t_quiet_min) + t_quiet_min;
        mydrive_left(t: t+dt_quite-1) = 0;
        t = t + dt_quite;
        dt_fire = randi(t_fire_max - t_fire_min) + t_fire_min;
        mydrive_left(t: t+dt_fire) = 1;
        t = t + dt_fire;
        
        t_right = t + randi(dt_max - dt_min) + dt_min;
        dt_fire_right = randi(t_fire_max - t_fire_min) + t_fire_min;
        mydrive_right(t_right:t_right+dt_fire_right) = 1;
        
    end
    
    mydrive_left = mydrive_left(1:ceil(tf + dn_info * nNodes/2));
    mydrive_right = mydrive_right(1:ceil(tf + dn_info * nNodes/2));
    
    mydrive = zeros(nNodes, tf);
    for ii = 1:nNodes/2
        mydrive(nNodes/2+1-ii,:) = ...
            mydrive_left(((ii-1)*dn_info+1):((ii-1)*dn_info+end-(nNodes/2)*dn_info));
        mydrive(nNodes+1-ii,:) = ...
            mydrive_right(((ii-1)*dn_info+1):((ii-1)*dn_info+end-(nNodes/2)*dn_info));
%         mydrive(nNodes/2+1-ii,:) = mydrive_left(ii:ii-1+end-(nNodes/2));
%         mydrive(nNodes+1-ii,:) = mydrive_right(ii:ii-1+end-(nNodes/2));
    end
    %%
    for itau = 1:ltau
    
        tau_ca = tau2(itau)
        params.tau_ca = tau_ca;
        
        f_sim = rand(nNodes, tf) < mydrive * spike_rate * dt_sim;
        
        f_sim_ca = zeros(nNodes, tf);
        for ii = 1:nNodes
            f_sim_ca(ii,:) = spike_to_calcium(f_sim(ii,:), params, 0.01);
        end
        
        f_sim_ca = f_sim_ca(:, t_burnin+1:end);
        f_sim = f_sim(:, t_burnin+1:end)+0;
        
        
        nSamples = size(f_sim_ca, 2);
%         f_sim_ca = f_sim_ca - repmat(mean(f_sim_ca,2),1,nSamples);
        
        %
        f_sim_ca = f_sim_ca(:, 1:floor(1/sampling_freq / dt_sim):end);
        
        %%

%         for maxLags = 1:4
        [~, bvgc_ca, mvgc_ca, bvgc_value_ca, mvgc_value_ca] = ...
                compute_corr_mat_and_gc_mat ...
                (f_sim_ca, maxLags, plotFlag, constantFlag);
        
        wipsi_sim(itau, 3, imc) = compute_ipsi_vs_contra(bvgc_value_ca, nl, nr);
        wipsi_sim(itau, 4, imc) = compute_ipsi_vs_contra(mvgc_value_ca, nl, nr);
        
        whtt_sim(itau, 3, imc) = ...
            compute_directional_preference(bvgc_value_ca, nl, nr);
        whtt_sim(itau, 4, imc) = ...
            compute_directional_preference(mvgc_value_ca, nl, nr);
%         figure(7)
%         subplot(4,2,(maxLags-1)*2+1)
%         imagesc(bvgc_value_ca)
%         colorbar()
%         axis square
% 
%         subplot(4,2,(maxLags-1)*2+2)
%         imagesc(mvgc_value_ca)
%         colorbar()
%         axis square
%         end

    end
end

end






