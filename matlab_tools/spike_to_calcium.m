function [sigma_noisy] = spike_to_calcium(sigma_sim, params, flnoise)
%  function [sigma_noisy] = spike_to_calcium(sigma_sim, params, flnoise)
% 
%  Transfers spiking data to calcium transients, convoluting the spikes 
%  with exponential decays.
%
%  Input:     1. sigma_sim: N * T matrix, with 0 and 1's giving spikes
%             2. params:             a structure, with the following fields
%                params.tau_ca:      the calcium decay time constant.
%                params.dt_sim:      time steps used in simulation.
%             3. flnoise (optional): a constant regulating the fluoresence 
%                                    noise
%
%  Output:    1. sigma_noisy: N * T matrix, the convoluted calcium
%                             transients
%

%%
[nNodes, lf] = size(sigma_sim);

tau_ca = params.tau_ca;
dt_sim = params.dt_sim; 

sigma_smooth = zeros(nNodes,lf);

for i = 1:nNodes
    s_spike = sigma_sim(i,:);
    s_smooth = 0;
    for t = 1:lf
        s_smooth = s_smooth + s_spike(t);
        s_smooth = s_smooth * exp(-dt_sim/tau_ca);
        sigma_smooth(i,t) = s_smooth;
    end
end

%% We set the base level of activity to 1.
sigma_smooth = sigma_smooth + 1;

%% Optional: add photon shot noise to the signal
if nargin < 3
    sigma_noisy = sigma_smooth;
else
    sigma_noisy = sigma_smooth + sqrt(sigma_smooth) .* randn(nNodes, lf) * flnoise ;
end

end
