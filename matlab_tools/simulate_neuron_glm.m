function [sigma_sim, sigma_sim_include_burnin, sigma_sim_sub] = ...
    simulate_neuron_glm(nNodes, tf, params, plotFlag)
% function [sigma_sim, sigma_sim_include_burnin, sigma_sim_sub] = ...
%    simulate_neuron_glm(nNodes, tf, params, plotFlag).
%
% Simulate the GLM dynamics for a group of neurons. 
%
% Input: 1. nNodes:         scalar, number of neurons to simulate
%        2. tf:             scalar, total number of time points to simulate
%        3. params:         structure, with fields
%           params.G:       nNodes * nNodes matrix, adjacency matrix
%           params.dt_sim:  time steps used in the simulation
%        4. plotFlag (optional): default = 0
%
% Ouput: 1. sigma_sim:      simulated data
%        2. sigma_sim_include_burnin: simulated data, include the time
%                                     points for the burn-in's
%        3. sigma_sim_sub:  sigma_sim, subsampled given the sampling time
%                           scale, params.tau_sample 
%

%%
if nargin < 3
    params.mu = ones(nNodes,1)*(-3);
    params.maxLags = 2;
    G = zeros(nNodes, nNodes, maxLags);
    G(:,:,1) = randn(nNodes)/sqrt(nNodes);
    G(:,:,1) = G(:,:,1) + eye(nNodes)*0.5; 
    G(:,:,2) = eye(nNodes)*(-0.25);
    params.G = G;
end

if nargin < 4
    plotFlag = 0;
end

%%
% Run the dynamics. Let's make sure that the memory depend only on the
% past, and that there is no equal time influence/couplings among pairs
% tf = 500;


switch params.method
    case 'stepwise'
        [sigma_sim, sigma_sim_include_burnin] = simulate_neuron_glm_stepwise_lag(params);
    case 'triangle'
        [sigma_sim, sigma_sim_include_burnin] = simulate_neuron_glm_triangle(params);
    case 'exponential'
        %%
        [sigma_sim, sigma_sim_include_burnin] = simulate_neuron_glm_exponential(params);
    otherwise
        G = params.G;
        mu = params.mu;
        maxLags = params.maxLags;
        dt_sim = params.dt_sim;
        tf = params.tf;
        lf = params.lf;
        
        sigma_sim = zeros(nNodes, tf);

        sigma0 = zeros(nNodes, maxLags); % start from zero spikes
        sigma_old = sigma0;
        
        for t = 1:tf
            sigma_sim(:,t) = sigma_old(:,end);
            loglambda = mu;
            for lag = 1:maxLags
                loglambda = loglambda + G(:,:,lag) * sigma_old(:,end-lag+1);
            end
            lambda = exp(loglambda);
            sigma_new_1t = (rand(nNodes,1)<lambda);
            sigma_new = [sigma_old(:,2:end) sigma_new_1t];
            sigma_old = sigma_new;
        end
end



if plotFlag
    figure(1)
    subplot(2,1,1)
    imagesc(sigma_sim)
end

if params.spike_subsample
    %%
    k = floor(params.tau_sample / params.dt_sim );
    lf = size(sigma_sim,2);
    sigma_sim_sub = reshape(sigma_sim(:,1:floor(lf/k)*k), nNodes, k, floor(lf/k));
    sigma_sim_sub = squeeze(any(sigma_sim_sub, 2));

    if plotFlag
        figure(1)
        subplot(2,1,2)
        imagesc(sigma_sim)
    end
else 
    sigma_sim_sub = sigma_sim;
end

end

function [sigma_sim, sigma_sim_include_burnin] = simulate_neuron_glm_stepwise_lag(params)

%%
nNodes = params.nNodes;
tau_info = params.tau_info;
tau_ref = params.tau_ref;
G0 = params.G;

tf = params.tf;
dt_sim = params.dt_sim;
t_burnin = params.t_burnin;


l_burnin = floor (t_burnin / dt_sim);
t = 0;
sigma_sim = zeros(nNodes, lf + l_burnin);
sigma_old = zeros(nNodes,1);
%

mu = -log(params.tau_spk);

count = 1;

%%
max_tlag = tau_info + tau_ref;
min_tlag = tau_info - tau_ref;
max_llag = floor(max_tlag / dt_sim);
min_llag = floor(min_tlag / dt_sim);


sigma_saved = zeros(nNodes, floor((tau_ref*2)/dt_sim+1));
llag = size(sigma_saved,2);

while t < tf + t_burnin
    sigma_sim(:, count) = sigma_old;
    loglambda = mu;
    
    if t > max_tlag
        sigma_saved = sigma_sim(:, count-max_llag:count-min_llag);
    end
    
    
    for lag = 1:llag
%         loglambda = loglambda + ...
%             G0 * sigma_saved(:,end-lag+1)/llag ;
%         loglambda = loglambda + ...
%             G0 * sigma_saved(:,end-lag+1)/((max_llag-min_llag+1)*dt_sim) ;
%            loglambda = loglambda + ...
%                 G0 * sigma_saved(:,end-lag+1)/((max_llag-min_llag+1)) ;
           loglambda = loglambda + ... 
                G0 * sigma_saved(:,end-lag+1)*dt_sim/tau_ref/2 ;
    end
    
    spike_probability = exp(loglambda) * dt_sim;
    sigma_new = (rand(nNodes,1)<spike_probability); % spikes
    
    
    t = t + dt_sim;
    count = count + 1;
    
    sigma_old = sigma_new;
    
%     if t > max_lag
%         sigma_saved = [sigma_saved(:,2:end) sigma_sim];
%     end
    
end


sigma_sim_include_burnin = sigma_sim;
sigma_sim = sigma_sim(:, (l_burnin + 1):end);
% disp(['count = ' num2str(count) ' sum(spike_1) = ' num2str(sum(sigma_sim(1,:)))])


end

function [sigma_sim, sigma_sim_include_burnin] = simulate_neuron_glm_triangle(params)

%%
nNodes = params.nNodes;
tau_info = params.tau_info;
tau_ref = params.tau_ref;
G0 = params.G;

tf = params.tf;
dt_sim = params.dt_sim;
t_burnin = params.t_burnin;
lf = floor(tf / dt_sim);
l_burnin = floor (t_burnin / dt_sim);
sigma_sim = zeros(nNodes, lf + l_burnin);
sigma_old = zeros(nNodes,1);
%

mu = -log(params.tau_spk);

max_tlag = tau_info + tau_ref;
min_tlag = tau_info - tau_ref;
max_llag = floor(max_tlag / dt_sim);
min_llag = floor(min_tlag / dt_sim);

llag = floor((tau_ref*2)/dt_sim+1);
if rem(llag,2) == 0
    llag = llag + 1;
end
sigma_saved = zeros(nNodes, llag);

count = 1;
t = 0;
while t < tf + t_burnin
    sigma_sim(:, count) = sigma_old;
    loglambda = mu;
    
    if t > max_tlag
        sigma_saved = sigma_sim(:, count-max_llag:count-min_llag);
    end
    
    triangle_coeff = (linspace(0,2,(max_llag-min_llag)/2+1)) ; 
    triangle_coeff = [triangle_coeff fliplr(triangle_coeff(1:end-1))];
    sigma_saved = sigma_saved .* repmat(triangle_coeff, nNodes, 1);
    
    
    for lag = 1:llag
        loglambda = loglambda + ...
            G0 * sigma_saved(:,end-lag+1)/((max_llag-min_llag+1)) ;
%         loglambda = loglambda + ...
%             G0 * sigma_saved(:,end-lag+1)/((max_llag-min_llag+1)*dt_sim) ;
    end
    
    spike_probability = exp(loglambda) * dt_sim;
    sigma_new = (rand(nNodes,1)<spike_probability); % spikes
    
    
    t = t + dt_sim;
    count = count + 1;
    
    sigma_old = sigma_new;
end

sigma_sim_include_burnin = sigma_sim;
sigma_sim = sigma_sim(:, (l_burnin+1):end);


end

function [sigma_sim, sigma_sim_include_burnin] = simulate_neuron_glm_exponential(params)

%%
nNodes = params.nNodes;
tau_info = params.tau_info;
G0 = params.G;

tf = params.tf;
dt_sim = params.dt_sim;
t_burnin = params.t_burnin;
lf = floor(tf / dt_sim);
l_burnin = floor (t_burnin / dt_sim);
t = 0;
sigma_sim = zeros(nNodes, lf + l_burnin);
sigma_old = zeros(nNodes,1);
%

mu = -log(params.tau_spk);
mem_kernel = zeros(nNodes,1);

count = 1;
while t < tf + t_burnin
    sigma_sim(:, count) = sigma_old;
    loglambda = mu;
    loglambda = loglambda + mem_kernel;
    spike_probability = exp(loglambda) * dt_sim;
    sigma_new = (rand(nNodes,1)<spike_probability); % spikes
    
    mem_kernel = mem_kernel * exp(-dt_sim/tau_info);
    mem_kernel = mem_kernel + (G0 * sigma_new) / tau_info * dt_sim;


    t = t + dt_sim;
    count = count + 1;
    
    sigma_old = sigma_new;
end


sigma_sim_include_burnin = sigma_sim;
sigma_sim = sigma_sim(:, (l_burnin+1):end);

end




