# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:39:22 2020

@author: Shaheen.Ahmed
"""
import numpy as np 
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from numpy.random import normal
from numpy.random import seed
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
from random import uniform
from random import seed
import Thesis_modules.All_thesis_functions.MSM as MSM
import Thesis_modules.All_thesis_functions.GPR_1_dim as GPR_1_dim
import Thesis_modules.All_thesis_functions.Calibration as Calibration


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)
    
def n_dim_GPR_in_sklearn_with_input_kernel(
                       input_kernel,
                       used_parameter_settings,
                       fitness_value_for_each_parameter_setting,
                       GPR_fit_title,
                       GPR_fit_filename,
                       index_of_sample_point_to_omit = None,
                       print_GPR_plot = False
                       ):
    '''
    This function implements GPR via sklearn, in any input dimension, and with any input kernel.
    
    Inputs:
    used_parameter_settings = (dimensionality x number_of_sample_points) numpy array, of all sample points in parameter space. 
    fitness_value_for_each_parameter_setting = fitness value for each parameter setting, averaged over repeated runs at each parameter setting.
    index_of_sample_point_to_omit = for CVPE purposes. Allows us to omit a sample point from the GPR fit, to get the error in our GPR fit. 
    Outputs:
    mu_test = GPR predictions at used_parameter_settings
    '''
    
    # Trim NaNs 
    # Repeated use, but should be OK. Only use in one place though?
    used_parameter_settings_with_nans_removed, fitness_array_with_nans_removed = Calibration.trim_nans_from_fitness_and_sample_points(used_parameter_settings, fitness_value_for_each_parameter_setting)    
    # Normalise fitness array, GPR seems to fail more often for high fitness values
    fitness_array_with_nans_removed = fitness_array_with_nans_removed / max(fitness_array_with_nans_removed)     
    # Transpose sample points to fit sklearn GPR code
    transposed_used_parameter_settings = used_parameter_settings_with_nans_removed.T   
    # Get indices of parameters which are varying. 
    # The locations where first row of transposed (first sample point) = second sample point, should be indices of non-varied points in param space.
    indices_of_varied_parameters = np.where(transposed_used_parameter_settings[0,:] != transposed_used_parameter_settings[1,:])[0]      
    # Remove constant parameters from parameter settings, for use later
    transposed_used_parameter_settings_only_varied_parameters = np.take(transposed_used_parameter_settings, indices_of_varied_parameters, axis = 1)  
    # Check if we are conducting CVPE or not, if not:
    
    if (index_of_sample_point_to_omit == None):        
        # Create GPR object with input kernel
        gpr = GaussianProcessRegressor(kernel = input_kernel)      
        # Fit GPR with only varying parameters, and fitness values (NaNs removed)
        # Note that while each column in our used_parameter_settings array has been a sample point, for gpr.fit, each row is a sample point.
        gpr.fit(transposed_used_parameter_settings_only_varied_parameters, fitness_array_with_nans_removed)      
        # Generate points to plot GPR prediction at. 
        prediction_points_varied_parameters = Calibration.generate_random_points_in_n_dim_hypercube(len(indices_of_varied_parameters), 
                                                                                       len(fitness_array_with_nans_removed)*20, 
                                                                                       np.amin(transposed_used_parameter_settings_only_varied_parameters, axis = 0), np.amax(transposed_used_parameter_settings_only_varied_parameters, axis = 0), np.empty(len(indices_of_varied_parameters)) * np.inf)[0].T # Improvement: remove indices_randomised_parameters, can get from zeros of vector_non_randomised_parameter_values.       
        # For GPR plotting, in one dimension, we need to sort the parameter values we'll be printing the GPR prediction at. 
        if (len(indices_of_varied_parameters) == 1):
            prediction_points_varied_parameters = np.sort(prediction_points_varied_parameters, axis = 0)        
        # Predict the GPR at prediction_points_varied_parameters
        mu_test, cov_test = gpr.predict(prediction_points_varied_parameters, return_cov=True)
        
        # If only one parameter varied (param space has dimensionality one)
        if (len(indices_of_varied_parameters) == 1):          
            # Plot the GPR at the given points
            GPR_1_dim.plot_gp(mu_test, 
                    cov_test, 
                    prediction_points_varied_parameters, 
                    transposed_used_parameter_settings_only_varied_parameters, 
                    fitness_array_with_nans_removed,
                    GPR_fit_title,
                    GPR_fit_filename)
            
        # If two parameters varied:
        if (len(indices_of_varied_parameters) == 2):                      
            # Get param values along first dimension
            first_dim_used_param_settings_with_fitness_nans_removed = used_parameter_settings_with_nans_removed[indices_of_varied_parameters[0]]            
            # Get param values along second dimension
            second_dim_used_param_settings_with_fitness_nans_removed = used_parameter_settings_with_nans_removed[indices_of_varied_parameters[1]]            
            # Plot 3d GPR
            plot_3d_GPR_figure(prediction_points_varied_parameters[:,0], prediction_points_varied_parameters[:,1], mu_test, first_dim_used_param_settings_with_fitness_nans_removed, second_dim_used_param_settings_with_fitness_nans_removed, fitness_array_with_nans_removed, FC_figure_filename, FC_figure_title)
        return mu_test
    
    # If we are conducting CVPE:
    else:        
        # Remove sample point i
        transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point = np.delete(transposed_used_parameter_settings_only_varied_parameters, index_of_sample_point_to_omit, axis = 0)       
        # Remove relevant fitness value
        fitness_array_with_nans_removed_without_omitted_sample_point = np.delete(fitness_array_with_nans_removed, index_of_sample_point_to_omit)      
        # Create GPR object with input kernel
        gpr = GaussianProcessRegressor(kernel = input_kernel)         
        # Fit GPR without sample point i
        gpr.fit(transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point, fitness_array_with_nans_removed_without_omitted_sample_point)         
        # Generate points to predict GPR at. 
        prediction_points_varied_parameters = Calibration.generate_random_points_in_n_dim_hypercube(len(indices_of_varied_parameters), 
                                                                                       len(fitness_array_with_nans_removed)*20, 
                                                                                       np.amin(transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point, axis = 0), np.amax(transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point, axis = 0), np.empty(len(indices_of_varied_parameters)) * np.inf)[0].T     
        # If only one parameter varied (param space has dimensionality one)
        if (len(indices_of_varied_parameters) == 1):
            prediction_points_varied_parameters = np.sort(prediction_points_varied_parameters, axis = 0)            
        # Fit without the omitted sample point (above), for CVPE, and predict on the full set of sample points. 
        mu_test, cov_test = gpr.predict(prediction_points_varied_parameters, return_cov=True)    
        # Fit without the omitted sample point (above), but also predict on the full set of sample points, to plot the prediction 
        mu_test_at_sample_points, cov_test_at_sample_points = gpr.predict(transposed_used_parameter_settings_only_varied_parameters, return_cov=True)       

        if (len(indices_of_varied_parameters) == 1):
            # Plot the GPR at the given points
            if (print_GPR_plot == True):
                GPR_1_dim.plot_gp(mu_test, 
                        cov_test, 
                        prediction_points_varied_parameters, 
                        transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point, 
                        fitness_array_with_nans_removed_without_omitted_sample_point,
                        GPR_fit_title,
                        GPR_fit_filename)
            
        # If two parameters varied:
        if (len(indices_of_varied_parameters) == 2):               
            # Get param values along first dimension
            first_dim_used_param_settings_with_fitness_nans_removed = transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point[:, 0]         
            # Get param values along second dimension
            second_dim_used_param_settings_with_fitness_nans_removed = transposed_used_parameter_settings_only_varied_parameters_without_omitted_sample_point[:, 1]        
            # Plot 3d GPR
            if (print_GPR_plot == True):
                plot_3d_GPR_figure(prediction_points_varied_parameters[:,0], prediction_points_varied_parameters[:,1], mu_test, first_dim_used_param_settings_with_fitness_nans_removed, second_dim_used_param_settings_with_fitness_nans_removed, fitness_array_with_nans_removed_without_omitted_sample_point, GPR_fit_title, GPR_fit_filename)     
        return mu_test_at_sample_points


def plot_3d_GPR_figure(first_dim_predicted_points, second_dim_predicted_points, predicted_points, first_dim_executed_param_values, second_dim_executed_param_values, fitness_values, GPR_fit_title, GPR_fit_filename):
    '''
    This function takes two sets of input points across two dimensions, points predicted by GPR, and actual execution points. 
    It also takes the values for these points, and plots the predicted points as a surface (very small matplotlib interpolation between many GPR predicted points), and 
    the executed points as a scatter plot.
    
    Inputs:
    first_dim_predicted_points = set of points GPR has produced predictions for, first dim. 
    second_dim_predicted_points = set of points GPR has produced predictions for, second dim. 
    predicted_points = values predicted by GPR at 2d points. 
    first_dim_executed_param_values = set of points function has been executed for, first dim. 
    secon_dim_executed_param_values = set of points function has been executed for, second dim. 
    fitness_values = values of function at 2d points.   
    Output:
    GPR surface and executed points. 
    '''
    
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(first_dim_predicted_points, second_dim_predicted_points, Z = predicted_points,cmap=cm.coolwarm, alpha = 0.5)
    ax.set_xlabel('First parameter value', fontsize=10)
    ax.set_ylabel('Second parameter value', fontsize=10)
    ax.set_zlabel('Fitness value', fontsize=10)
    ax.scatter3D(first_dim_executed_param_values, second_dim_executed_param_values, fitness_values, color = "green")
    plt.title(GPR_fit_title)
    plt.savefig(GPR_fit_filename)
    plt.show()
