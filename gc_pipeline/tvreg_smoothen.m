function tvreg_smoothen(fn, sampling_frequency)

zs_f = load([fn '_zs.txt']);

[nNodes, nSamples] = size(zs_f);

%%

smooth_f = nan(size(zs_f));
noise_power2 = nan(1,nNodes);
acc_power2 = nan(1,nNodes);

for i = 1:nNodes
    my_noisy_f = zs_f(i,:);
    [my_smooth_f, ~, ~, optimal_alpha, noise_power, acc_power] = ...
        extractSmoothDerivative(my_noisy_f, sampling_frequency);
    smooth_f(i,:) = my_smooth_f; 
    noise_power2(i) = noise_power;
    acc_power2(i) = acc_power;
end

%% 
fn3 = [fn '_snr.txt'];
fp3 = fopen(fn3, 'w');
for i = 1:nNodes
    fprintf(fp3, '%.6f %.6f \n', noise_power2(i), acc_power2(i));
end

fclose(fp3);
%% 

fn2 = [fn '_smooth.txt'];
fp2 = fopen(fn2, 'w');
for i = 1:nNodes
    for t = 1:nSamples
        fprintf(fp2, '%.6f ', zs_f(i,t));
    end
    fprintf(fp2,'\n');
end
fclose(fp2);

%%
figure(1)
subplot(1,2,1)
imagesc(corr(zs_f'))
axis square
xlabel('Neuron ID')
ylabel('Neuron ID')
colorbar()
title('Corr(f)')

subplot(1,2,2)
imagesc(corr((zs_f - smooth_f)'))
axis square
xlabel('Neuron ID')
ylabel('Neuron ID')
colorbar()
title('Corr(noise)')

saveas(gcf,'corr_noise.pdf')

end



