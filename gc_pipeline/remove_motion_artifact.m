function remove_motion_artifact(fn)

%%
noisy_f = load([fn '.txt']);

[nNodes, nSamples] = size(noisy_f);


%% Step 1: remove motion artifact

ma_corrected_f = nan(size(noisy_f));

for i = 1:nNodes
    ma_corrected_f(i,:) = filloutliers(noisy_f(i,:),'linear','movmedian',5);
end


%%
fn2 = [fn '_ma_corrected.txt'];
fp2 = fopen(fn2, 'w');
for i = 1:nNodes
    for t = 1:nSamples
        fprintf(fp2, '%.6f ', ma_corrected_f(i,t));
    end
    fprintf(fp2,'\n');
end
fclose(fp2);

end