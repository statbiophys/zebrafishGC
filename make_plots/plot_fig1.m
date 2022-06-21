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


%
figure(3)
plot(sigma_sim_glm_cont(1,:)+10)
hold on
plot(sigma_sim_glm_cont(4,:)+5)
plot(sigma_sim_glm_cont(10,:)-4)
hold off
xlim([200 300])

%%
figure(4)
subplot(3,1,1)
stem(sigma_sim_glm(1,:))
xlim([200 300])

subplot(3,1,2)
stem(sigma_sim_glm(4,:), 'color', matlab_red)
xlim([200 300])

subplot(3,1,3)
stem(sigma_sim_glm(10,:), 'color',matlab_yellow)
xlim([200 300])
        
        
end
