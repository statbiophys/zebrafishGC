# zebrafishGC
Python and MATLAB code for Granger causality analysis on empirically observed calcium transients in zebrafish neuronal populations, and on synthetic dynamics generated on toy networks. This set of codes is assciated with the manuscript "Granger causality analysis for calcium transients in neuronal networks: challenges and improvements", by X. Chen, F. Ginoux, T. Mora, A. M. Walczak, and C. Wyart (2022). The data used in the paper, and can be used as sample data for this pipeline, can be found https://zenodo.org/record/6774389#.Yr2JvC2l2S5

Most functions have both a MALAB and a python version.

The pipeline for analyzing the hindbrain data is written in python, and can be found in gc_pipeline/demo_GC.ipynb. All other steps of the pipeline written in python can be found in the folder gc_pipeline.

To smooth noisy data using total-variation regularization, please refer to the MATLAB code matlab_tools/extractSmoothDerivative.m. 

The MATLAB code to generate synthetic data can be find in folders matlab_plots. All auxillary functions written in MATLAB, including simulating dynamics, measuring GC, and computing the directional preference, are included in the folder matlab_tools.

Last updated: 2022. June. 30
