function [smooth_f, df, disc_f, optimal_alpha, noise_power, acc_power] = ...
    extractSmoothDerivative(noisy_f, sampling_frequency, hardThres, plotFlag)
%
%
% [smooth_f, df, disc_f] = ...
%             extractSmoothDerivative (noisy_f, sampling_frequency)
%
% Goal:
% 1. Given 1d noisy (temporal) signal, estimate its derivative.
% 2. Based on the derivative, discretize into n states.
%  
% Note:
% 1. Default number of states for discretization is 3
% 2. This program assumes white noise.
% 3. This code refer to TVRegDiff.m, developed by Rick Chartrand,
%          , which you can download here:
%          https://sites.google.com/site/dnartrahckcir/home/tvdiff-code
% 4. This code was first created for X.Chen, F.Randi, A.M.Leifer, 
%        W.Bialek, Physical Review E 99 (5), 052418 (2019)
%
%
% Input:      noisy_f                   - 1D array of noisy signal
%             sampling_frequency        - unit: frames per second
%             hardThres (optional       - threshold for discretization,
%                                         default = 5
%             plotFlag (optional)       - default = 0/off
%
% Output:     disc_f                    - 1D array of derivative-based
%                                         discrete signal                                     
%             smooth_f                  - 1D array of smooth signal, 
%                                         anti-derivative of df
%             df                        - 1D array of derivative, estimated
%                                         using total-variational
%                                         regularization
%             optimal_alpha             - optimal value for the alpha
%                                         parameter (Ref[2])
%             noise_power               - power of noise
%             acc_power                 - total power of signal + noise
%
% Reference: 1. R.Chartrand, ISRN Applied Mathematics 2011
%            2. X.Chen, F.Randi, A.M.Leifer, W.Bialek, 
%               Physical Review E 99 (5), 052418 (2019)
%
% Xiaowen Chen
% 2017-12-07 (edited 2018-11-15)
% All rights reserved
%
%%
%
% This software is a 1D discretizationer, published under GPLv3 license
% Copyright (C) 2018 Xiaowen Chen.
%
% This program is a free software: you can redistribute it and/or modify 
% it under the terms of the GNU General Public License as published by 
% the Free Software Foundation, either version 3 of the License, or 
% (at your option) any later version. 

% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% General Public License for more details.

%%
% 0. Set default values for input arguments
if nargin < 4
    plotFlag = 0;
end

if nargin < 3
    hardThres = 5;
end

if nargin < 2
    sampling_frequency = 6;
end

if nargin < 1
    load test.mat test
    noisy_f = test;
end

% Initialize
l = size(noisy_f);
nStates = 3;


%%
% 1. Exclude nan from the data. Will put back after smoothing.
[noisy_f_partial, good_idx] = exclude_nan(noisy_f);

%%
% 2. Use TV regularization to find smooth df and smooth f

wn_floor_init = 250;

[smooth_f_partial, df_partial, optimal_alpha, noise_power, acc_power] = ...
    smooth_tvreg(noisy_f_partial, sampling_frequency, plotFlag, wn_floor_init);

smooth_f = nan(l);
smooth_f(good_idx) = smooth_f_partial;
df = nan(l);
df(good_idx) = df_partial;

%%
% 3. Discretize the input signal using a hard threshold
df = fillmissing(df,'linear');

disc_f = ones(l)*nStates;

signal_power = acc_power - noise_power;

df_thres = hardThres * noise_power.^(3/2)/sampling_frequency/...
    optimal_alpha/signal_power^(1/2) ;

disc_f(df>df_thres) = 1;
disc_f(df<-df_thres) = 2;

%%
% 4. visualize
if 1 %plotFlag
    figure(10)
    subplot(2,1,1)
    plot((1:length(noisy_f))/sampling_frequency, noisy_f, 'color', [0 0 0]+0.65)
    hold on
    plot((1:length(noisy_f))/sampling_frequency, smooth_f)
    hold off
    set(gca,'fontsize',12)
    xlabel('Time (s)')
    ylabel('f')
    
    subplot(2,1,2)
    for iState = 1:nStates
        plot(find(disc_f==iState)/sampling_frequency, ...
            smooth_f(disc_f==iState), '.')
        hold on
    end
    hold off
    set(gca,'fontsize',12)
    xlabel('Time (s)')
    ylabel('f')

end

end


function [output, good_index] = exclude_nan(input)
% exclude nan from the data

output = input(~isnan(input));
good_index = find(~isnan(input));

% make the length of the signal an even number, for FFT purpose
if rem(length(good_index),2) ~= 0 
    output = output(1:end-1);
    good_index = good_index(1:end-1);
end

end



function [smooth_signal, deriv_signal, optimal_alpha, noise_power, acc_power] = ...
    smooth_tvreg(input, sampling_frequency, plotFlag, wn_floor_init)
% given a signal that could contain nan's, but otherwise has white noise,
% smooth it with TV regularization by picking out the alpha that maximize
% the similarity between white noise and the residue after we smooth out
% the noise

% Among the outputs: noise_power = power of the noise; acc_power = power of
% the noise plus power of the signal
if nargin < 4
    wn_floor_init = 500;
end

if nargin < 3
    plotFlag = 0;
end

if nargin < 2
    sampling_frequency = 6;
end

%%
nSamples = length(input);

% 1.1 extract power of the signal
[p1,f] = plot_power_spectrum(input, sampling_frequency, plotFlag);

power_s = p1(2:end);
acc_power = cumsum(power_s,'reverse');
%%
% 1.2 extrapolate the white noise floor for the signal
fit_param=polyfit(f(wn_floor_init+1:end),acc_power(wn_floor_init:end),1);
noise_floor = f(2:end)*fit_param(1)+fit_param(2);
noise_power = noise_floor(1);

if plotFlag
    figure(4)
    plot(f(2:end), acc_power)
    hold on
    plot(f(2:end), noise_floor)
end

acc_power = acc_power(1);

%%
% 2.1 apply Total-variation regularization to smoothen the derivative
% scan through alpha, which is a combination of parameters in TV reg,
% see R. Chartrand (2011) for details
alphaSet = 2.^(-10:1:3);
[tvsmoothstruct] = initialize_tvreg(nSamples, sampling_frequency);
tvsmoothstruct.nsamples = length(input);
tvsmoothstruct.gridSpacing = 1/sampling_frequency;
tvsmoothstruct.plotFlag = plotFlag;

errorMaster = zeros(1,length(alphaSet));

% 2.2 minimize the MSE between the noise extrapolated from the Fourier transform
% of the raw signal and the noise induced by alpha in a total variational
% regularization
my_function = @(my_alpha)tvreg_noisediffwhite(my_alpha,...
    noise_floor,tvsmoothstruct,input, 1, sampling_frequency);


for ialpha = 1:length(alphaSet)
    errorMaster(ialpha) = my_function(alphaSet(ialpha));
%     pause(0.1);

end

%%
% 3.1 Zoom in on alpha, given errorMaster

[~,min_error_alpha_index] = min(errorMaster);

alpha_min_error = alphaSet(max(min_error_alpha_index-1,1));
alpha_max_error = alphaSet(min(min_error_alpha_index+1,length(alphaSet)));
if min_error_alpha_index == 1
    alpha_min_error = alphaSet(1)/2;
elseif min_error_alpha_index == length(alphaSet)
    alpha_max_error = alphaSet(end)*2;
end


alphaSet2 = 2.^(linspace(log2(alpha_min_error),log2(alpha_max_error),14));

% 3.2 Scan through the zoomed in alphaSet, find the alpha that gives the
% minimum error

for ialpha = 1:length(alphaSet2)

    errorMaster(ialpha) = my_function(alphaSet2(ialpha));
%     pause(0.1)

end

[~,min_error_alpha_index2] = min(errorMaster);
optimal_alpha = alphaSet2(min_error_alpha_index2);

%%
[~,smooth_signal, deriv_signal] = ...
    my_function(alphaSet2(min_error_alpha_index2));

end

function [my_error, smooth_signal, deriv_signal] = ...
    tvreg_noisediffwhite(my_alpha, extrapolated_acc_power_noise, ...
    tvsmoothstruct, rawsig, alpha_optimal2, sampling_frequency)
% Return the MSE between the noise extrapolated from the Fourier transform
% of the raw signal and the noise induced by alpha in a total variational
% regularization

%% 1. Correct for the orientation of signal
[m,n] = size(rawsig);
if m < n
    rawsig = rawsig';
end

[m,n] = size(extrapolated_acc_power_noise);
if m < n
    extrapolated_acc_power_noise = extrapolated_acc_power_noise';
end


%%
Fs = sampling_frequency; 
T = 1/Fs; % sampling period
L = tvsmoothstruct.nsamples; % number of time points
t = (0:L-1)*T; % time
%%

plotFlag = tvsmoothstruct.plotFlag;

extendWindow = 200;
constWindow = 50;

if plotFlag
    figure(1)
end

tvsmoothstruct.alpha = alpha_optimal2 * my_alpha;
tvsmoothstruct.gridSpacing = T;
smoothSignal1 = tvsmooth(rawsig,tvsmoothstruct);

% estimate the noise, by subtracting smooth f from noisy f
dfMaster1 = rawsig - smoothSignal1;
stddfset = nanstd(dfMaster1);

% correct for edge effect, by extending the noisy signal using the std
% extracted from (smooth f - noisy f)
extendsig = zeros(1,length(rawsig)+extendWindow*2);
extendsig(extendWindow+1:end-extendWindow) = rawsig;
edgeConst = nanmedian(rawsig(1:constWindow));
extendsig(1:extendWindow) = ...
    edgeConst + stddfset * randn(1,extendWindow);
edgeConst = nanmedian(rawsig(end-constWindow+1:end));
extendsig(end-extendWindow+1:end) = ...
    edgeConst+stddfset*randn(1,extendWindow);

% smooth again, after extending the noisy signal
tvsmoothstruct.nsamples = L + extendWindow*2;
[smooth_signal2,deriv_signal2] = tvsmooth(extendsig,tvsmoothstruct);

% chop off extra time point
smooth_signal = smooth_signal2(extendWindow+1:end-extendWindow);
deriv_signal = deriv_signal2(extendWindow+1:end-extendWindow);

% estimate the noise again, by subtracting smooth f from noisy f
dfMaster2 = rawsig - smooth_signal;

% find the power of the noise, dfMaster2
Y = fft(dfMaster2);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f  = Fs*(0:(L/2))/L;
power_noise = P1(2:end).^2;

acc_power_noise = cumsum(power_noise,'reverse')/2;

% Compute error between the noise (extracted by TV regularization) and the
% noise (extracted by white noise floor estimation)

my_error = ...
    sqrt(nanmean((acc_power_noise-extrapolated_acc_power_noise).^2))*10000;
% * 10000 to make the error large, for fminunc purpose

if plotFlag
    figure(2)
    plot(f(2:end),acc_power_noise,'linewidth',2)
    hold on
    plot(f(2:end),extrapolated_acc_power_noise,'linewidth',2)
    hold off
    legend('smooth noise', 'true noise')
end


end

function [P1, f] = plot_power_spectrum(data, sampling_frequency, plotFlag)
% plot the one-sided power spectrum of time sequence using discrete Fourier
% transform

if nargin < 3
    plotFlag = 0;
end

if nargin < 2
    sampling_frequency = 6; % sampling frequency = 6Hz
end

%%
T = 1/sampling_frequency; % sampling period
L = length(data);
t = (0:L-1)*T;


% apply Fast Fourier Transformation
Y = fft(data);
P2 = abs(Y/L).^2; 
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = sampling_frequency*(0:(L/2))/L;


if plotFlag
    figure(1)
    % Power spectrum graph
    plot(f,P1,'linewidth',1.5) 
    set(gca,'xscale','log');
    set(gca,'fontsize',14);
    title('Single-Sided Amplitude Spectrum of X(t)')
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    
    figure(2)
    % Cumulant sum of power spectrum
    plot(f(2:end),cumsum(P1(2:end),'reverse'),'linewidth',1.5)
end

end


function [smooth_f, df] = tvsmooth(noisy_f,tvsmoothstruct)
% wrapper for TV regularization

if nargin < 2
    nIter = 10;
    alpha = 1;
    ep = 1e-6; % epsilon to avoid dividing by 0
    gridSpacing = 1/6;
    diagFlag = 0;
    plotFlag = 0;
else
    nIter = tvsmoothstruct.nIter;
    alpha = tvsmoothstruct.alpha;
    ep = tvsmoothstruct.ep;
    gridSpacing = tvsmoothstruct.gridSpacing;
    diagFlag = tvsmoothstruct.diagFlag;
    plotFlag = tvsmoothstruct.plotFlag;
end


nSamples = length(noisy_f);


if plotFlag
    figure(1)
    subplot(2,1,1)
    plot((1:nSamples)*gridSpacing, noisy_f);

    subplot(2,1,2)
end


[df, antiDeriv] = TVRegDiff(noisy_f, nIter, alpha, [], 'small', ep, ...
    gridSpacing, plotFlag, diagFlag);

smooth_f = antiDeriv(df)+noisy_f(1);


df = (df(1:end-1)+df(2:end))/2; % gives value at the original time points


if plotFlag
    figure(1)
    subplot(2,1,2);
    hold on
    plot((1:nSamples)*gridSpacing-gridSpacing/2,df')
    hold off
    set(gca,'fontsize',12);
    xlabel('Time (s)')
    ylabel('d(\Delta R/R_0)/dt')
    axis([gridSpacing/2 nSamples*gridSpacing-gridSpacing/2 ...
        min(df) max(df)])

    subplot(2,1,1);
    
    
    hold on
    plot((1:nSamples-1)*gridSpacing,antiDeriv(df)+noisy_f(1),'linewidth',2);
    hold off
    set(gca,'fontsize',12);
    xlabel('Time (s)')
    ylabel('\Delta R/R_0')
    axis([gridSpacing/2 nSamples*gridSpacing-gridSpacing/2 ...
        min(noisy_f) max(noisy_f)])
end

end

function [tvsmoothstruct] = initialize_tvreg(nSamples, sampling_frequency)
% initialize total-variation regularization method

nIter=10;
ep=1e-6;
gridSpacing=1/sampling_frequency;
diagFlag=0;
plotFlag=0;
extendWindow=200;
constWindow=50;
tvsmoothstruct=struct('alpha',1,'nIter',nIter,'ep',ep,...
    'gridSpacing',gridSpacing,'diagFlag',diagFlag,'plotFlag',plotFlag,...
    'nsamples',nSamples,'extendWindow', extendWindow, ...
    'constWindow', constWindow);



end
