# zebrafishGC
Python and MATLAB code for Granger causality analysis on empirically observed calcium transients in zebrafish neuronal populations, and on synthetic dynamics generated on toy networks. This set of codes is assciated with the manuscript "Granger causality analysis for calcium transients in neuronal networks: challenges and improvements", by X. Chen, F. Ginoux, T. Mora, A. M. Walczak, and C. Wyart (2022). The data used in the paper, and can be used as sample data for this pipeline, can be found https://zenodo.org/record/6774389#.Yr2JvC2l2S5

The complete pipeline, which can take user input data in .txt format, is in gc_pipeline/run_pipeline.sh. 
Example data is included in the folder, as "example.txt" (for 20 neurons in a hindbrain dataset) and "example_mt.txt" (for motoneuron in fish f3t2). 
The bash script run_pipeline.sh calls various MALAB and python functions, which are all included in the folder gc_pipeline.

For complete GC analysis as described in the manuscript, please find demo_GC_hindbrain.ipynb and demo_GC_motoneuron.ipynb as codes that analyzes the data presented in the paper, also in the folder gc_pipeline.. 

The MATLAB code to generate synthetic data can be find in folders matlab_plots. All auxillary functions written in MATLAB, including simulating dynamics, measuring GC, and computing the directional preference, are included in the folder matlab_tools.

----

In data_paper folder: Pandas DataFrame and Python dictionnaries
For hindbrain and for motoneurons

to load data: 
import pickle
pickle_filename = 'YOUR_DATA_PATH/df_name.pkl'  # change accordingly
with open(pickle_filename, 'rb') as pickle_in:
     df_name = pickle.load(pickle_in)

pickle_filename = 'YOUR_DATA_PATH/dict_name.pkl' 
with open(pickle_filename, 'rb') as pickle_in:
    dict_name = pickle.load(pickle_in)  




Motorneuron data for three examples: 
Fish 3 Trial 1 and Fish 5 Trial 2 for Figure 3.
Fish 5 Trial 2 for figure 4.

Columns:
- Fish: fish index
- Trace: trace index
- fluo: fluorescence traces [n_cells x n_timesteps]
- fluo_type: 'dff' or 'f_smooth', respectively before and after smoothing procedure 
	     (bad neurons remmoved, no motion artifact correction for dff)
- n_cells: number of cells in the plane (only those kept for analysis, "bad" cells removed)
- mid: middle cell, to split left vs right neurons (left until index mid-1, right from index mid and on)
- cell_centers: x and y position of the cells [n_cells x 2]
- multivariate: boolean to indicate bivariate (False) or multivariate (True) GC
- GC: Granger causality matrix results [n_cells x n_cells]
- GC_sig: Granger causality matrix results, significant with original threshold (where Fstat > threshold_F) [n_cells x n_cells]
- GC_sig_new_thresh: Granger causality matrix results, significant with new threshold (where Fstat > new_threshold_F) [n_cells x n_cells]
- Fstat: F-statistics matrix [n_cells x n_cells]
- threshold_F: original threshold for the F-statistics significance
- new_threshold_F: new threshold for the F-statistics after the whole pipeline is applied

Dictionaries with data for all fish-trace pairs:
to get data from a dictionary: dict_name.get((fish, trace))
- artifact_dict.pkl: artifact index
- background_dict.pkl: image of cells (mean of calcium imaging recording)
- cell_centers_dict.pkl: x and y position of all cells
- cell_centers_removed_dict.pkl: x and y position of cells excluding "bad" neurons
- middle_dict.pkl: middle neuron among all cells to separate left-right sides 
- middle_removed_dict.pkl: middle neuron among cells excluding "bad" neurons to separate left-right sides 
- dff_dict_with_bad_neurons.pkl: original fluorescence traces, for all cells
- dff_removed_dict.pkl: original fluorescence traces, without "bad" neurons
- dff_corrected_dict.pkl: original fluorescence traces, without "bad" neurons, without motion artifacts
- dff_smoothed_dict.pkl: smoothed fluorescence traces, without "bad" neurons, without motion artifacts
- new_threshold_dict.pkl: customized threshold.


Hindbrain data
Fish 6 Trial 07

Columns:
- fluo: fluorescence traces [n_cells x n_timesteps]
- cell_centers: x and y position of the cell center [n_cells x 2]
- background: plane background for plotting [249 x 512]
- n_cells: number of cells in the plane
- tail_angle: array of angle of the tail [75000,] - 75000 timesteps: higher frequency than calcium imaging recording
- tail_angle_regressor: tail angle convolved to calcium decay function [75000,] 
- is_swim: boolean array whether each cell in correlated to swim activity (True if pearson correlation between cell's fluorescence trace and tail_angle_regressor > 0.6) [n_cells,]
- swim_neurons: indices of swim-correlated neurons [n_swim_cells,]
- medial_neurons: indices of swim-correlated neurons in the medial region [n_medial_cells,]
- only_swim_neurons: indices of swim-correlated neurons NOT in the medial region [n_only_swim_cells,]
- SNR: signal-to-noise ratio for each cell [n_cells,]

- BV_GC_all: original bivariate (BV) Granger causality results matrix on all cells [n_cells,n_cells]
- BV_GC_medial: original bivariate (BV) Granger causality results matrix [n_medial_cells,n_medial_cells]
- BV_Fstat_medial: original BV F-statistics matrix [n_medial_cells,n_medial_cells]
- BV_threshold_F_ori_all: original threshold for the BV F-statistics significance for GC on all cells.
- BV_threshold_F_ori: original threshold for the BV F-statistics significance for GC on medial cells only.
	/!\  BV_threshold_F_ori and BV_threshold_F_ori_all different due to Bonferroni correction
- BV_threshold_F_new_mat_medial: new threshold customized for each pair of neurons (BV) [n_medial_cells,n_medial_cells]
- BV_Fstat_normalized_medial: new BV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
- BV_GC_normalized_medial: new BV GC results matrix normalized by customized threshold [n_medial_cells,n_medial_cells]

- MV_GC_medial: original multivariate (MV) Granger causality results matrix [n_medial_cells,n_medial_cells]
- MV_Fstat_medial: original MV F-statistics matrix [n_medial_cells,n_medial_cells]
- MV_threshold_F_ori_medial: original threshold for the MV F-statistics significance
- MV_threshold_F_new_mat_medial: new MV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
- MV_Fstat_normalized_medial: new MV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
- MV_GC_normalized_medial: new MV GC results matrix normalized by customized threshold [n_medial_cells,n_medial_cells]


To run demo faster, use data present in these files.
To do the processing step-by-step, follow the guidelines in the notebooks. This takes a lot of time.
Note that the smoothing needs to be done in Matlab (see matlab_code folder).

----
Last updated: 2022. July. 3
