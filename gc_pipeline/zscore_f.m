function zscore_f(fn)

%%
ma_corrected_f = load([fn '_ma_corrected.txt']);

[nNodes, nSamples] = size(ma_corrected_f);


%% Step 1: remove motion artifact

zs_f = nan(size(ma_corrected_f));

for i = 1:nNodes
    zs_f(i,:) = (ma_corrected_f(i,:) - mean(ma_corrected_f(i,:))) ...
        / std(ma_corrected_f(i,:)) ;
end


%%
fn2 = [fn '_zs.txt'];
fp2 = fopen(fn2, 'w');
for i = 1:nNodes
    for t = 1:nSamples
        fprintf(fp2, '%.6f ', zs_f(i,t));
    end
    fprintf(fp2,'\n');
end
fclose(fp2);

end