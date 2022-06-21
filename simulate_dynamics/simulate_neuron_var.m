function [sigma_sim, sigma_sim_include_burnin, sigma_sim_sub] = ...
    simulate_neuron_var(nNodes, tf, params, plotFlag)
% function [sigma_sim, sigma_sim_include_burnin, sigma_sim_sub] = ...
%    simulate_neuron_var(nNodes, tf, params, plotFlag)
%
% Generate synthetic neuronal dynamics using Vector Autoregressive Model
%
% Input: 1. nNodes:         scalar, number of neurons to simulate
%        2. tf:             scalar, total number of time points to simulate
%        3. params:         structure, with fields
%           params.G:       nNodes * nNodes matrix, adjacency matrix
%           params.maxLags: true lags;
%           params.noise_std: add white noise to the VAR
%        4. plotFlag (default = 0): if 1, plot figures. 
%
% Ouput: 1. sigma_sim:      simulated data
%        2. sigma_sim_include_burnin: simulated data, include the time
%                                     points for the burn-in's
%        3. sigma_sim_sub:  sigma_sim, subsampled given the sampling time
%                           scale, params.tau_sample. 
%

%%
if nargin < 4
    plotFlag = 0;
end

if nargin < 3
    params.mu = ones(nNodes,1)*(-3);
    params.maxLags = 2;
    G = zeros(nNodes, nNodes, params.maxLags);
    G(:,:,1) = randn(nNodes)/sqrt(nNodes);
    G(:,:,1) = G(:,:,1) - eye(nNodes)*0.5; 
    G(:,:,2) = eye(nNodes)*(-0.25);
    params.G = G;
    params.noise_std = 0.02;
    plotFlag = 1;
end


%%
G = params.G;

maxLags = params.maxLags;
noise_std = params.noise_std; 

%%
sigma_sim = zeros(nNodes, tf);

sigma0 = zeros(nNodes, maxLags); 
sigma_old = sigma0;

for t = 1:tf
    sigma_sim(:,t) = sigma_old(:,end);
    sigma_new_1t = zeros(nNodes,1); % params.mu
    for lag = 1:maxLags
        sigma_new_1t = sigma_new_1t + G(:,:,lag) * sigma_old(:,end-lag+1);
    end
    sigma_new_1t = sigma_new_1t + randn(nNodes,1) * noise_std;

    sigma_new = [sigma_old(:,2:end) sigma_new_1t];
    sigma_old = sigma_new;
end

if plotFlag
        %
    figure(1)
    plot(sigma_sim(1,:))
    hold on
    plot(sigma_sim(2,:))
    plot(sigma_sim(3,:))
    hold off


    %%
    figure(2)
    cc = corr(sigma_sim');
    imagesc(cc)
    axis square
    colorbar()

    %%
    figure(3)
    crosscorr(sigma_sim(1,:), sigma_sim(2,:))

end


end



