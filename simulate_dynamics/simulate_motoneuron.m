function simulate_motoneuron

%%
 addpath(genpath('~/MyDocuments/biophysics/projects/maxent/misctools'))
%% 
nNodes = 10;


sampling_freq = 4 ; % 4 hz
params.dt_sim = 1/sampling_freq;

t_fire_min = 7; % 8 time points continuously
t_fire_max = 15;
t_quiet_min = 20; %30; 
t_quiet_max = 50;

spike_rate = 1;% 0.8;

dt_min = 2; 
dt_max = 10; 

maxLags = 1;


%
tf = 1000;
t_burnin = 1000;
tf = tf + t_burnin;



%%
tau2 = 0.5:0.5:10; %0.5:0.5:5; %1:20; %[1:5:21]; %[1,2,4,8,16,32,64];
ltau = length(tau2);
nmc = 10;
wipsi_sim = zeros(ltau, 4, nmc);
whtt_sim = zeros(ltau, 4, nmc);
nl = 5; nr = 5;


% %%

constantFlag = 0;
plotFlag = 0;


for imc = 1:nmc
    imc = imc

    %%
    mydrive_left = zeros(1, tf + nNodes/2);
    mydrive_right = zeros(1, tf + nNodes/2);
    
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
    
    mydrive_left = mydrive_left(1:tf + nNodes/2);
    mydrive_right = mydrive_right(1:tf + nNodes/2);
    
    mydrive = zeros(nNodes, tf);
    for ii = 1:nNodes/2
        mydrive(nNodes/2+1-ii,:) = mydrive_left(ii:ii-1+end-(nNodes/2));
        mydrive(nNodes+1-ii,:) = mydrive_right(ii:ii-1+end-(nNodes/2));
    end
    
    
    
    %%

%%
    for itau = 1:ltau
        
        tau_ca = tau2(itau)
        params.tau_ca = tau_ca;
    
        f_sim = rand(nNodes, tf) < mydrive * spike_rate;
    
        f_sim_ca = zeros(nNodes, tf);
        for ii = 1:nNodes
            f_sim_ca(ii,:) = spike_to_calcium(f_sim(ii,:), params, 0.01);
        end
        
        f_sim_ca = f_sim_ca(:, t_burnin+1:end);
        f_sim = f_sim(:, t_burnin+1:end)+0;
    
    
        nSamples = size(f_sim_ca, 2);
        f_sim_ca = f_sim_ca - repmat(mean(f_sim_ca,2),1,nSamples);
    
        [~, bvgc_ca, mvgc_ca, bvgc_value_ca, mvgc_value_ca] = ...
        compute_corr_mat_and_gc_mat ...
        (f_sim_ca, maxLags, plotFlag, constantFlag);
    
    
        
        [~, bvgc_spk, mvgc_spk, bvgc_value_spk, mvgc_value_spk] = ...
            compute_corr_mat_and_gc_mat ...
            (f_sim, maxLags, plotFlag, constantFlag);
    
    
        wipsi_sim(itau, 1, imc) = compute_ipsi_vs_contra(bvgc_value_spk, nl, nr);
        wipsi_sim(itau, 2, imc) = compute_ipsi_vs_contra(mvgc_value_spk, nl, nr);
        wipsi_sim(itau, 3, imc) = compute_ipsi_vs_contra(bvgc_value_ca, nl, nr);
        wipsi_sim(itau, 4, imc) = compute_ipsi_vs_contra(mvgc_value_ca, nl, nr);
    
    
        whtt_sim(itau, 1, imc) = ...
            compute_directional_preference(bvgc_value_spk, nl, nr);
        whtt_sim(itau, 2, imc) = ...
            compute_directional_preference(mvgc_value_spk, nl, nr);
        whtt_sim(itau, 3, imc) = ...
            compute_directional_preference(bvgc_value_ca, nl, nr);
        whtt_sim(itau, 4, imc) = ...
            compute_directional_preference(mvgc_value_ca, nl, nr);


        %%
%         if imc == 1
%             figure(7)
%             plot((1:1000)/4,f_sim_ca([1,6],:)');
%             pause
%         end

    end

end

%%
python_green = [80, 161, 37]/256;
python_purple = [115, 26, 162]/256;

figure(3)
errorbar(tau2, mean(wipsi_sim(:,3,:), 3), ...
    std(wipsi_sim(:,3,:), 0, 3),...
    '-o','color',python_purple)
hold on
errorbar(tau2, mean(wipsi_sim(:,4,:), 3), ...
    std(wipsi_sim(:,4,:), 0, 3), ...
    '-d','color',python_green)
plot([0 max(tau2)],[0.5 0.5],'--','color',[0 0 0]+0.65)
plot([0 max(tau2)],[1 1],'k-.')
plot([2.5 2.5],[0 1.5],'r')
hold off
xlabel('\tau_{ca} (s)')
ylabel('W_{ipsi}^{sim.}')
set(gca,'fontsize',12)
% ylim([-0.1 1.1])
ylim([0.3 1.2])
yticks([0.5 1])
l=legend('BVGC','MVGC');
l.Location='southeast';

figure(4)
errorbar(tau2, mean(whtt_sim(:,3,:), 3), ...
    std(whtt_sim(:,3,:), 0, 3), ...
    '-o','color',python_purple)
hold on
errorbar(tau2, mean(whtt_sim(:,4,:), 3), ...
    std(whtt_sim(:,4,:), 0, 3), ...
    '-d','color',python_green)
plot([0 max(tau2)],[0.5 0.5],'--','color',[0 0 0]+0.65)
plot([0 max(tau2)],[1 1],'k-.')
plot([2.5 2.5],[0 1.5],'r')

hold off
xlabel('\tau_{ca} (s)')
ylabel('W_{rc}^{sim.}')
set(gca,'fontsize',12)
yticks([0.5 1])

% ylim([-0.1 1.1])
ylim([0.3 1.2])


%% Plot the spikes and tau_ca
mydrive_left = zeros(1, tf + nNodes/2);
mydrive_right = zeros(1, tf + nNodes/2);

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

mydrive_left = mydrive_left(1:tf + nNodes/2);
mydrive_right = mydrive_right(1:tf + nNodes/2);

mydrive = zeros(nNodes, tf);
for ii = 1:nNodes/2
    mydrive(nNodes/2+1-ii,:) = mydrive_left(ii:ii-1+end-(nNodes/2));
    mydrive(nNodes+1-ii,:) = mydrive_right(ii:ii-1+end-(nNodes/2));
end




%%




tau_ca = 10;
params.tau_ca = tau_ca;

f_sim = rand(nNodes, tf) < mydrive * spike_rate;

figure(3)
subplot(2,1,1)
imagesc(f_sim)
colormap([1 1 1; 0 0 0])
xlim([200 600])
xticks([])
set(gca,'fontsize',12)
%

f_sim_ca = zeros(nNodes, tf);
for ii = 1:nNodes
    f_sim_ca(ii,:) = spike_to_calcium(f_sim(ii,:), params, 0.01);
end

f_sim_ca = f_sim_ca(:, t_burnin+1:end);
f_sim = f_sim(:, t_burnin+1:end)+0;

figure(3)
subplot(2,1,2)
plot(f_sim_ca')
xlim([200 600])
set(gca,'fontsize',12)
xlabel('t (s)')


% plot(f_sim_ca')

% %
% figure(4)
% constantFlag = 0;
% [~, bvgc_block, mvgc_block, bvgc_value_block, mvgc_value_block] = ...
%     compute_corr_mat_and_gc_mat ...
%     (mydrive, maxLags, 1, constantFlag);


% save('simulate_motoneuron_for_fig.mat');

end



