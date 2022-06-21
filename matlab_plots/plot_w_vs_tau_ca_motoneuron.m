function plot_w_vs_tau_ca_motoneuron (dt_info)
% Simulate zebrafish embryonic motoneurons' calcium transients
% with empirically observed statistics of the data
%
% Compute the directional preference W_{IR} and W_{RC} as a function of
% calciumd decay time scale tau_{ca}, at the information propagation time 
% scale, the input dt_{info}.
% 
% We want to check if GC can learn the information flow when 
% tau_info < tau_sampling
% In the manuscript, we plot for both tau_info = 0.25s and tau_info = 0.025s,
% both less than or equal to tau_sampling = 0.25s.
%
% Fix tau_ca = 2.5s, tau_sampling = 0.25s, as given by the experiment
% lambda = 32 s^-1
%
% This code plot Fig 5 B.


nNodes = 10;

sampling_freq = 4 ; 

dt_sim = 0.0125; 
% dt_info = 0.025; %0.25;
dn_info = floor(dt_info / dt_sim);
params.dt_sim = dt_sim;

t_fire_min = 8 / sampling_freq / dt_sim; 
t_fire_max = 16 / sampling_freq / dt_sim ;
t_quiet_min = 20 / sampling_freq / dt_sim ; 
t_quiet_max = 50 / sampling_freq / dt_sim ;

spike_rate =  32; 

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
tau2 = (0.5:0.5:10); 
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
        
        wipsi_sim(itau, 1, imc) = compute_ipsi_vs_contra(bvgc_value_ca, nl, nr);
        wipsi_sim(itau, 2, imc) = compute_ipsi_vs_contra(mvgc_value_ca, nl, nr);
        
        whtt_sim(itau, 1, imc) = ...
            compute_directional_preference(bvgc_value_ca, nl, nr);
        whtt_sim(itau, 2, imc) = ...
            compute_directional_preference(mvgc_value_ca, nl, nr);
    end
end

%%
figure(1)
subplot(1,2,1)
plot(tau2, mean(wipsi_sim(:,1,:),3))
hold on
plot(tau2, mean(wipsi_sim(:,2,:),3))
hold off
legend('BVGC','MVGC')
ylabel('W_{IC}')
xlabel('\tau_{ca} (s)')
title(['\tau_{info} = ' num2str(dt_info) 's'])

subplot(1,2,2)
plot(tau2, mean(whtt_sim(:,1,:),3))
hold on
plot(tau2, mean(whtt_sim(:,2,:),3))
hold off
legend('BVGC','MVGC')
ylabel('W_{RC}')
xlabel('\tau_{ca} (s)')
title(['\tau_{info} = ' num2str(dt_info) 's'])

end






