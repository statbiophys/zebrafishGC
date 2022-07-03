#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
from scipy import stats
from scipy.stats import zscore, pearsonr, mannwhitneyu
from random import sample
import matplotlib.pyplot as plt
from common_functions import *
from plotting_functions import *


filename = input("Enter file name containing the fluorescent traces (include .txt).\n");
print("Load " + filename + "...");
signals = np.loadtxt(filename);
#signals = np.loadtxt("example.txt");

sampling_frequency = input("Enter sampling frequency in Hz (for hindbrain data, 5.81Hz; for motoneurondata, 4Hz):\n")
sampling_frequency = float(sampling_frequency);

print('Plot example neural trace...')
plt.plot(signals[1])
plt.show(block=False)
plt.pause(1)
input("Press Enter to continue ...")
plt.close()

(n_cells, n_timesteps) = signals.shape
mid = int(n_cells/2)
print('Number of cells = %d.' % n_cells)
print('Number of timesteps = %d.' % n_timesteps)
n_pairs = n_cells * (n_cells - 1)

n_lags = 3
while True:
    n_lags = input("Enter number of lags (default value = 3): \n")
    try:
        n_lags = int(n_lags)
    except ValueError:
        print('Lags need to be a positive integer')
        continue
    if n_lags > 0:
        break

p_val = 0.01
while True:
    p_val = input("Enter p-value for significant test of GC (default = 0.01): \n")
    try:
        p_val = float(p_val)
    except ValueError:
        print('p-value need to be a positive number from 0 to 1')
        continue
    if 0 < p_val < 1:
        break
    else:
        print('p-value need to be a positive number from 0 to 1')


condition_flag = input('Condition on the stimulus ? (y/n)\n')
while True:
    if condition_flag == 'y':
        cdFlag = 1;
        stim_filename = input('Enter filename for stimulus (include .txt)\n')
        print('Load stimulus: ' + {stim_filename} + '...')
        stim = np.loadtxt(stim_filename)
        break
    elif condition_flag == 'n':
        cdFlag = 0;
        break
    else:
        print('Please enter y/n')

condition_flag = cdFlag


bvgc_flag = input('Enter 1 if compute bivariate Granger causality. Enter 0 if compute multivariate Granger causality.\n')

bvgc_flag = int(bvgc_flag)



if ~condition_flag:
    if bvgc_flag:
        print('Compute bivariate Granger causality...')        
        gc_sig, gc, fstat_gc, threshold_f = bivariateLinearCausalityTE(signals)
    else:
        print('Compute multivariate Granger causality...')
        gc_sig, gc, fstat_gc, threshold_f = multivariateLinearCausalityTE(signals);
else:
    if bvgc_flag:
        print('Compute bivariate Granger causality, conditioned on stimulus ...')
        gc_sig, gc, fstat_gc, threshold_f = conditionedLinearCausalityTE(signals, stim)
    else:
        print('Compute multivariate Granger causality, conditioned on stimulus ...')
        gc_sig, gc, fstat_gc, threshold_f = conditionedLinearCausalityTE(signals, stim, multi=True)
    

plt.figure()
plot_gc_matrix(gc, mid)
if bvgc_flag:
    plt.savefig('bvgc.png', dpi=200)
    print('BVGC matrix is plotted, saved to bvgc.png.')
else:
    plt.savefig('mvgc.png',dpi=200)
    print('MVGC matrix is plotted, saved to mvgc.png')

plt.show(block=False)
plt.pause(1)
input("Press Enter to continue ...")
plt.close()


### Generate adaptive GC threshold using shuffled data
print('Now, we generate null data by cyclically shuffling the input signal, and visualize the distribution of F-stats')
nmc = input('Enter the number of Monte Carlo shuffles (default = 1000, start with 20 for test):\n' )
print('nmc = ' + nmc)
nmc = int(nmc)

periodic_stim = input('Is the stimulus regular / periodic ? (Enter y/n)\n')
while True:
    if periodic_stim == 'n':
        periodic_stim_flag = 0
        break
    elif periodic_stim == 'y':
        periodic_stim_flag = 1
        stimT = input('Enter the period of the stimuli (in seconds) (for hindbrain data, stim_T = 15s):\n')
        stimT = float(stimT)
        
        stim_start = input('Enter the starting time of the first stimuli (in seconds) (for hindbrain data, stim_start = 5s):\n')
        stim_start = float(stim_start)
        
        nStim = input('Enter the total number of repeats of the stimuli (for hindbrain data, nStim = 19):\n')
        nStim = int(nStim)
        t_cycles = [round(sampling_frequency*(stim_start + i*stimT)) for i in range(nStim+1)]
        break
    else:
        print('Please enter y/n')

        
all_gcs_sh = np.zeros([n_cells, n_cells, nmc])
all_fstats_sh = np.zeros([n_cells, n_cells, nmc])

for i in range(n_cells):
    for j in range(n_cells):
        print('Receiver neuron ID i = %d, driver neuron ID j = %d' % (i, j))
        if i != j:
            signal2 = signals[j]
            Z_indices = list(set(range(n_cells))-set([i,j]))
            Z_set = signals[Z_indices]

                        
            for imc in range(nmc):
                if periodic_stim_flag:
                    signal1_sh = shift_signal(signals[i], t_cycles)
                else:
                    signal1_sh = shift_signal(signals[i])

                if bvgc_flag:
                    _, gc_imc, fstat_imc, _ = bvgc_2_signals(signal1_sh,
                                                         signal2,
                                                         n_lags=n_lags,
                                                         pval = p_val)
                else:
                    _, gc_imc, fstat_imc, _ = mvgc_2_signals(signal1_sh,
                                                             signal2, Z_set,
                                                             n_lags=n_lags,
                                                             pval=p_val)

                all_gcs_sh[i][j][imc] = gc_imc
                all_fstats_sh[i][j][imc] = fstat_imc
                
        else:
            for imc in range(nmc):
                all_gcs_sh[i][j][imc] = np.nan
                all_fstats_sh[i][j][imc] = np.nan
                
all_fstats_sh_nonzero = all_fstats_sh[all_fstats_sh>0]


plt.figure()
plt.hist(all_fstats_sh.flatten()[all_fstats_sh.flatten()>0],
         density=True, bins=600)
x = np.linspace(0,10,100)
dfn = n_lags

if bvgc_flag:
    dfd = n_timesteps - 3 * n_lags - 1
else:
    dfd = n_timesteps - (n_cells+1)*n_lags - 1
    
plt.plot(x, stats.f.pdf(x, dfn, dfd), 'r-',
         lw=5, alpha=0.6, label='f pdf')
plt.xlim([0,10])
plt.savefig('fstat_histogram.png', dpi=200)
print('Plot histogram for f-statistics, computed from cyclically shuffled data')
plt.show(block=False)
plt.pause(1)
input("Press Enter to continue ...")
plt.close()



# Check if the distribution follows an f-distribution

# fit the empirical f-stat distribution from shuffled data to a continuous distribution

while True:
    pair_specific_null = input('Do you want to apply pairwise-specific significance test? (y/n)\n If yes, a normalized GC will be computed. If no, the f-stat distribution from all pairs will be used to compute a significance threshold.\n Choose yes for hindbrain data, choose no for motoneuron data.')
    if pair_specific_null == 'y':
        pair_specific_flag = 1
        break
    elif pair_specific_null == 'n':
        pair_specific_flag = 0
        break
    else:
        print('Please enter y/n.')


### Pair-specific null distribution:
if pair_specific_flag:
    print('\n Here, we demonstrate the simplest null distribution observed in the hindbrain data for pair-specific f-stats, which is a scaled F-distribution, how to correct for the GC value. In reality, when you are using the pipeline, please find a proper distribution for your own data. You can start trying with scaled F-distributions with different degrees of freedom.\n')
    mean_f_sh = np.nanmean(all_fstats_sh, axis=2)
    all_fstats_sh_rescaled = np.zeros([n_cells, n_cells, nmc])

    for i in range(n_cells):
        for j in range(n_cells):
            if i != j:
                for imc in range(nmc):
                    all_fstats_sh_rescaled[i,j,imc] = all_fstats_sh[i,j,imc] / mean_f_sh[i,j]
                    
                else:
                    for imc in range(nmc):
                        all_fstats_sh_rescaled[i,j,imc] = np.nan

                        
    threshold_f_new = mean_f_sh * threshold_f
    fstat_normalized = fstat_gc / mean_f_sh

    if bvgc_flag:
        gc_normalized = get_GC_from_Fstat(fstat_normalized, 2*n_lags+1, n_lags+1, n_timesteps - n_lags)
    else:
        gc_normalized = get_GC_from_Fstat(fstat_normalized, n_cells*n_lags+1, (n_cells-1)*n_lags+1, n_timesteps - n_lags)
    gc_normalized_sig_new = get_GC_sig(gc_normalized, fstat_gc, threshold_f_new)

else:
    print('For non-pair specific GC, we demonstrate using the motoneuron data, where the empirical distribution of the f-stats for shuffled data falls on f-distribution with different degrees of freedom. This keeps the GC matrix the same, but changes the threshold for GC values to be significant. Please refer to demo_GC_motoneuron.ipynb for the details')

    print('Computing likelihood of the shuffled f-stats, for f-distributions with adjusted degrees of freedom...')

    outlier_threshold = input('Enter f-stat threshold to exclude outliers (default = 20, enter 0 if no threshold)')
    outlier_threshold = float(outlier_threshold)
    
    log_likelihood_list = []
    for dfd_to_fit in range(1,21):
        log_like = 0
        for my_fstat in all_fstats_sh_nonzero:
            if outlier_threshold > 0:
                if my_fstat < outlier_threshold:
                    log_like = log_like + np.log(stats.f.pdf(my_fstat, n_lags, dfd_to_fit))
            else:
                log_like = log_like + np.log(stats.f.pdf(my_fstat, n_lags, dfd_to_fit))

        log_like = log_like / len(all_fstats_sh_nonzero)
        log_likelihood_list.append(log_like)

    dfd_fit = np.argmax(log_likelihood_list)
    threshold_new = stats.f.ppf(1 - p_val/n_pairs, n_lags, dfd_fit)
    gc_sig_new_threshold = get_GC_sig(gc, fstat_gc, threshold_new)


### visualize the adjusted gc
print('Visualize adjusted GC')
if pair_specific_flag:
    plt.figure()
    plot_gc_matrix(gc_normalized,mid)
    plt.savefig('gc_normalizded.png',dpi=200)
    print('Normalized GC, adjusted pair-specifically, saved to gc_normalized.png')
    plt.show(block=False)
    plt.pause(1)
    input("Press Enter to continue ...")
    plt.close()
else:
    plt.figure()
    plot_gc_matrix(gc_sig_new_threshold,mid)
    plt.savefig('gc_sig_new_threshold.png',dpi=200)
    print('GC matrix with only significant links, using data-adjusted significance threshold, saved to gc_sig_new_threshold.png')
    plt.show(block=False)
    plt.pause(1)
    input("Press Enter to continue ...")
    plt.close()


print("save the GC matrix ...")
np.savetxt('gc.out', gc, fmt='%1.4e')
np.savetxt('gc_sig.out', gc_sig, fmt='%1.4e')
if pair_specific_flag:
    np.savetxt('gc_normalized.out', gc_normalized, fmt='%1.4e')
    np.savetxt('gc_normalized_sig.out', gc_normalized_sig_new, fmt='%1.4e')
else:
    np.savetxt('gc_sig_new_threshold.out', gc_sig_new_threshold, fmt='%1.4e')
    
    
print("End of the pipeline.")
quit()
