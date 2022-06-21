function simulate_motoneuron_w_vs_tau_info
% XC 2022-04-22
%
% Fix tau_ca = 2.5s, tau_sampling = 0.25s
% lambda = 32 s^-1
% 
% Vary tau_info from 0.01s to 0.25s, see how accurate is W_{IR} W_{RC}

%%
% nNodes = 10;

sampling_freq = 4 ; % 4 hz
% params.dt_sim = 1/sampling_freq;

dt_sim = 0.0125; %0.025; %0.125;
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

tau_ca = 2.5 ; % unit is second

%%
dt_info2 = 0.0125:0.0125:0.25;
ltau_info = length(dt_info2);
wipsi_sim = zeros(ltau_info, 2, nmc);
whtt_sim = zeros(ltau_info, 2, nmc);



%%
for itau = 1:ltau_info
    dt_info = dt_info2(itau);
    dn_info = floor(dt_info / dt_sim);

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
python_green = [80, 161, 37]/256;
python_purple = [115, 26, 162]/256;
figure(1)
errorbar(dt_info2, mean(wipsi_sim(:,1,:),3),...
    std(wipsi_sim(:,1,:),0,3),'-o','color',python_purple)
hold on
errorbar(dt_info2, mean(wipsi_sim(:,2,:),3),...
    std(wipsi_sim(:,2,:),0,3),'-d','color',python_green)
hold off
ylabel('W_{ipsi}^{sim}')
set(gca,'fontsize',12)
xlim([0 0.25])
xlabel('\tau_{info} (s)')

figure(2)
errorbar(dt_info2, mean(whtt_sim(:,1,:),3),...
    std(whtt_sim(:,1,:),0,3),'-o','color',python_purple)
hold on
errorbar(dt_info2, mean(whtt_sim(:,2,:),3),...
    std(whtt_sim(:,2,:),0,3),'-d','color',python_green)
hold off
ylabel('W_{RC}^{sim}')
xlabel('\tau_{info} (s)')
set(gca,'fontsize',12)
xlim([0 0.25])
yticks([0.5 1])



end



