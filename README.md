# zebrafishGC
Python and MATLAB code for Granger causality analysis on empirically observed calcium transients in zebrafish neuronal populations, and on synthetic dynamics generated on toy networks. This set of codes is assciated with the manuscript "Granger causality analysis for calcium transients in neuronal networks: challenges and improvements", by X. Chen, F. Ginoux, C. Wyart, T. Mora, and A. M. Walczak (2022).


The pipeline for analyzing the hindbrain data can be found in gc_pipeline/demo_GC.ipynb

To smooth noisy data using total-variation regularization, please refer to the MATLAB code in tvrg_smooth.

The MATLAB code to generate synthetic data can be find in folders simulate_dynamics, and make_plots. All auxillary functions are included in the folder matlab_tools.

Last updated: 2022. June. 21
