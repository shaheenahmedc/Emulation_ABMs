# All_thesis_functions
A repository containing all code used in my Masters thesis at UU. 

AR_1.py:
| Function        | Notebook created in           | Description | 
| ------------- |:-------------:| -----:|
| generate_ar1_series     | First experiments - AR(1)  | Outputs an AR(1) time series. |
| plot_ar1_series     | First experiments - AR(1)  | Plots an AR(1) time series. |

Brock_Hommes.py:
| Function        | Notebook created in           | Description | 
| ------------- |:-------------:| -----:|
| generate_Brock_Hommes_time_series     | Brock_Hommes Experiments  | Outputs time series from '98 Brock_Hommes model |

CVPE.py:
| Function        | Notebook created in           | Description |
| ------------- |:-------------:| -----:|
| calculate_and_plot_cvpe_1d_manual_gpr     | First experiments - AR(1)  | Implements CVPE from Barde_17, in one dimension, for our manual GPR implementation |
| calculate_and_plot_cvpe_1d_sklearn_gpr     | First experiments - AR(1)  | Implements CVPE from Barde_17, in one dimension, for our first sklearn GPR implementation |
| calculate_and_plot_cvpe_1d_gpy_gpr     | First experiments - AR(1)  | Implements CVPE from Barde_17, in one dimension, for our first GPy GPR implementation |
| calculate_and_plot_cvpe_n_dim_sklearn_gpr     | Brock_Hommes Experiments  | Implements CVPE from Barde_17, in n dimensions, with sklearn. Prints GPR fit at each iteration, differences between actual MSM values and predicted, and a final CVPE value. Visualisation works for one and two dimensions | 


GPR_1_dim.py:
| Function        | Notebook created in           | Description |
| ------------- |:-------------:| -----:|
| kernel     | First experiments - AR(1)  | Implements squared exponential kernel in one dimension manually |
| plot_gp     | First experiments - AR(1)  | Plots a GPR surface, over a given range, for a set of x_train and y_train points |
| plot_covariance_heatmap     | First experiments - AR(1)  | Plots covariance matrix in one dimension situations |
| GP_posterior_mean_and_cov     | First experiments - AR(1)  | Manual implementation of posterior mu(x) and cov(x) vectors, given x_train and y_train| 
| nll_fn     | First experiments - AR(1)  | Manual calculation of negative log-likelihood, for input to minimisation function. Optimise kernel hyperparameters|
| GPR_wrapper     | First experiments - AR(1)  | Wrap manual GPR process, inputting training data and outputting a plot of one-dimensional GPR surface|
| GPR_wrapper_sklearn_sq_exp_kernel_plus_noise     | First experiments - AR(1)  | Wrap initial sklearn GPR implementation|
| GPR_wrapper_gpy_sq_exp_kernel_plus_noise     | First experiments - AR(1)  | Wrap initial GPy GPR implementation|

GPR_n_dim.py:
| Function        | Notebook created in           | Description |
| ------------- |:-------------:| -----:|
| plot_gp_2D     | First experiments - AR(1)  | Initial 2d plotting, probably redundant now|
| n_dim_GPR_in_sklearn_with_input_kernel     | Brock_Hommes Experiments   | This function implements GPR via sklearn, in any input dimension, and with any input kernel. Also takes (d x n) parameter setting used, and (n) MSM values. Outputs GPR predictions at parameter settings.|
| plot_3d_GPR_figure     | Brock_Hommes Experiments  | Plots GPR surface for 2d emulation problem, as well as true MSM surface.|

MSM.py:
| Function        | Notebook created in           | Description |
| ------------- |:-------------:| -----:|
| generate_moments_for_MSM     | First experiments - AR(1)  | Generate moments for MSM, between model-generated time series vs pseudo_true/empirical.|
| apply_weighting_matrix_to_moments     | First experiments - AR(1)   | Apply W matrix in MSM (identity or simple weighting matrix) |
| MSM_wrapper     | First experiments - AR(1)  | Takes inputs of two time series, outputs MSM value|
| calculate_and_plot_param_settings_vs_MSM_fn_1d     | First experiments - AR(1)  | Plots parameter values against MSM distance function between pseudo_true and model-generated, in one dimension, but only for AR(1) currently|
| calc_and_plot_MSM_empirical_vs_model_generated     | First experiments - AR(1)  | Plots param values against MSM (empirical vs model-generated), one-dim, AR(1)|
| trim_nans_from_MSM_and_sample_points     | Brock_Hommes Experiments  | Removes NANs from MSM, as well as the relevant sample points. Brock_Hommes can diverge, resulting in some infinite values|
| calculate_and_plot_param_settings_vs_MSM_fn_n_dim     | Brock_Hommes Experiments  | Plots (1 or 2 dim) param values vs MSM from pseudo-true/empirical time series, for n parameters. Pass parameters to vary as np.inf. See notebook for implementation.|
| generate_random_points_in_n_dim_hypercube     | Brock_Hommes Experiments  | Generate sample points in n-dim space.|
| plot_3d_MSM_figure     | Brock_Hommes Experiments | Plot 3d MSM surface (not GPR implementation, just matplotlib interpolation), from two input vectors, and MSM values|
| plot_1_dim_MSM     | Brock_Hommes Experiments  | Plot 1d MSM results|

Sandtable_Inheritance.py:
| Function        | Notebook created in           | Description |
| ------------- |:-------------:| -----:|
| run_SA_on_Sandtable_Inheritance     | Sandtable_Inheritance - Experiments  | Run SA on SI model. To be generalised. Extract model run function and sampling functionality|
| run_SI_model     | Sandtable_Inheritance - Experiments  | Run SI model through command line magic. Input parameter array and redundant time series parameter to use with calculate_and_plot_param_settings_vs_MSM_fn_n_dim|

Sensitivity_Analysis.py:
| Function        | Notebook created in           | Description |
| ------------- |:-------------:| -----:|
| plot_1_dim_SA     | Sandtable_Inheritance - Experiments  | Plot results of 1-dim sensitivity analysis. Very similar to MSM plotting functions, if anything more simple. |
