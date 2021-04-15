# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:41:00 2020

@author: Shaheen.Ahmed
"""

import numpy as np 
import matplotlib.pyplot as plt
from numpy.random import seed
import Thesis_modules.All_thesis_functions.MSM as MSM
import Thesis_modules.All_thesis_functions.GPR_n_dim as GPR_n_dim
import Thesis_modules.All_thesis_functions.Calibration as Calibration
from matplotlib import cm


def calculate_and_plot_cvpe_n_dim_sklearn_gpr(
                                     used_parameter_settings,
                                     fitness_value_for_each_sample_point,
                                     input_kernel_with_hyperparam_intialisations, 
                                     GPR_fit_filename,
                                     GPR_fit_title,
                                     CVPE_filename,
                                     CVPE_title,
                                     CVPE_error_filename,
                                     CVPE_error_title
                                     ):
    '''
    This function implements the cross-validation prediction error method of Barde and van der Hoog 2017, with sklearn.
    If we're sampling a one or two dimensional space, it prints the GPR fit at each iteration, and the final differences between the GPR prediction at each parameter setting, and the actual value of the function at that parameter setting. 
    It also outputs the final cvpe value. 
    
    Inputs:
    used_parameter_settings = (d x n) numpy array of used parameter settings. d = number of rows, also number of parameters. n = number of columns, also number of sample points. NOTE: Opposite to what sklearn requires. 
    fitness_value_for_each_sample_point = fitness value for each parameter setting, averaged over repeated runs at each parameter setting. 
    input_kernel_with_hyperparam_intialisations = An sklearn kernel with initial conditions for the optimisation of the sklearn kernel hyperparameters. 
    
    Outputs:
    cvpe = final CVPE value.
    '''
    
    # Should trim NaNs here be hashed out, as it's currently done in run_n_dim_frequentist_calibration. How to guarantee it's always done?
    # No harm in running again, unless slow?
    used_parameter_settings_with_nans_removed, fitness_array_with_nans_removed = Calibration.trim_nans_from_fitness_and_sample_points(used_parameter_settings, fitness_value_for_each_sample_point) 
    
    # Normalise fitness
    # When to do, always at start?
    fitness_array_with_nans_removed = fitness_array_with_nans_removed / max(fitness_array_with_nans_removed)

    # Initialise CVPE sum
    cvpe_sum = 0
    
    # Get the number of sample points from the used parameter settings
    number_of_sample_points = len(used_parameter_settings_with_nans_removed[0,:])
    
    # Initialise array to hold estimated GPR values for each sample point i 
    Estimated_fitness_for_i_via_GPR_for_plotting = np.empty(number_of_sample_points)
    
    # Loop over sample points
    for i in range(number_of_sample_points): 
        print (f'GPR CVPE loop is at {i}')
        if (i == int(len(range(number_of_sample_points)) / 2)):
            Estimated_fitnesses_via_GPR = GPR_n_dim.n_dim_GPR_in_sklearn_with_input_kernel(input_kernel_with_hyperparam_intialisations, used_parameter_settings_with_nans_removed, fitness_array_with_nans_removed, GPR_fit_title, GPR_fit_filename, index_of_sample_point_to_omit = i, print_GPR_plot = True)
        else:
            Estimated_fitnesses_via_GPR = GPR_n_dim.n_dim_GPR_in_sklearn_with_input_kernel(input_kernel_with_hyperparam_intialisations, used_parameter_settings_with_nans_removed, fitness_array_with_nans_removed, GPR_fit_title, GPR_fit_filename, index_of_sample_point_to_omit = i, print_GPR_plot = False)
        # Run GPR on the I-O pairs, without i.
        # Provide index of sample point i to n_dim_GPR_in_sklearn_with_input_kernel. It will then know we're conducting CVPE, and produce the relevant GPR estimates. 
        # Note, we've predicted on all sample points, but fit on all sample points excluding i 

        # Store estimated fitness for i 
        Estimated_fitness_for_i_via_GPR_for_plotting[i] = Estimated_fitnesses_via_GPR[i] 
        #  Get the actual fitness for i 
        actual_fitness_for_i = fitness_array_with_nans_removed[i] 
        
        # Calculate squared difference between actual fitness and GPR estimate
        squared_diff_between_actual_and_estimated_fitness = (actual_fitness_for_i - Estimated_fitnesses_via_GPR[i])**2
        
        # Add to CVPE total
        cvpe_sum += squared_diff_between_actual_and_estimated_fitness

    # Divide the CVPE sum by the number of sample points
    cvpe = cvpe_sum/len(used_parameter_settings)
    
    # Get the indices of varied parameters
    # The locations where first row of transposed (first sample point) = second sample point, should be indices of non-varied points in param space. 
    indices_of_varied_parameters = np.where(used_parameter_settings_with_nans_removed[:,0] != used_parameter_settings_with_nans_removed[:,1])[0] 

    # If one parameter is being varied:
    if (len(indices_of_varied_parameters) == 1):
        
        first_dim_used_param_settings_with_fitness_nans_removed = used_parameter_settings_with_nans_removed[indices_of_varied_parameters[0]]
        # Fitness arrays needs to be sorted, but according to the param values they were calculated for
        Estimated_fitness_for_i_via_GPR_for_plotting = Estimated_fitness_for_i_via_GPR_for_plotting[first_dim_used_param_settings_with_fitness_nans_removed.argsort()]
        fitness_array_with_nans_removed = fitness_array_with_nans_removed[first_dim_used_param_settings_with_fitness_nans_removed.argsort()]
        
        first_dim_used_param_settings_with_fitness_nans_removed = first_dim_used_param_settings_with_fitness_nans_removed[first_dim_used_param_settings_with_fitness_nans_removed.argsort()]

        # Plot Estimated vs actual GPR fits
        plt.figure(figsize = (10,7))
        plt.plot(first_dim_used_param_settings_with_fitness_nans_removed, Estimated_fitness_for_i_via_GPR_for_plotting,  label = 'Estimated')    
        plt.plot(first_dim_used_param_settings_with_fitness_nans_removed, fitness_array_with_nans_removed, label = 'Actual')
        plt.xlabel('Parameter value')
        plt.ylabel('Fitness value')
        plt.legend()
        plt.title(CVPE_title)
        plt.savefig(CVPE_filename)

        # Plot differences between actual and estimated GPR fits
        plt.figure(figsize = (10,7))
        plt.plot(first_dim_used_param_settings_with_fitness_nans_removed, (Estimated_fitness_for_i_via_GPR_for_plotting - fitness_array_with_nans_removed))
        plt.xlabel('Parameter value')
        plt.ylabel('CVPE')
        plt.legend()
        plt.title(CVPE_error_title)
        plt.savefig(CVPE_error_filename)
        plt.show()
        
    # If two parameters varied:
    elif (len(indices_of_varied_parameters) == 2):
       
        # Get parameter values for first dimension
        first_dim_used_param_settings_with_fitness_nans_removed = used_parameter_settings_with_nans_removed[indices_of_varied_parameters[0]]
        # Get parameter values for second dimension
        second_dim_used_param_settings_with_fitness_nans_removed = used_parameter_settings_with_nans_removed[indices_of_varied_parameters[1]]
        
        # Create 3d fitness plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot estimated GPR fitness
        surf_1 = ax.plot_trisurf(first_dim_used_param_settings_with_fitness_nans_removed, second_dim_used_param_settings_with_fitness_nans_removed, Z = Estimated_fitness_for_i_via_GPR_for_plotting, cmap=cm.Blues, label = 'Estimated', facecolor = 'b', alpha = 0.5)
        
        # Plot actual fitnesses
        surf_2 = ax.plot_trisurf(first_dim_used_param_settings_with_fitness_nans_removed, second_dim_used_param_settings_with_fitness_nans_removed, Z = fitness_array_with_nans_removed, cmap=cm.Reds, label = 'Actual', facecolor = 'r', alpha = 0.5)       
        surf_1._facecolors2d=surf_1._facecolors3d
        surf_1._edgecolors2d=surf_1._edgecolors3d
        surf_2._facecolors2d=surf_2._facecolors3d
        surf_2._edgecolors2d=surf_2._edgecolors3d      
        ax.set_xlabel('First parameter value', fontsize=10)
        ax.set_ylabel('Second parameter value', fontsize=10)
        ax.set_zlabel('Fitness value', fontsize=10)
        ax.legend(loc = 'lower right')
        plt.title(CVPE_title)
        plt.savefig(CVPE_filename)

        # Plot estimated - actual 
        difference_array = (fitness_array_with_nans_removed - Estimated_fitness_for_i_via_GPR_for_plotting)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(first_dim_used_param_settings_with_fitness_nans_removed, second_dim_used_param_settings_with_fitness_nans_removed, Z = difference_array, cmap=cm.coolwarm, alpha = 0.5)
        ax.set_xlabel('First parameter value', fontsize=10)
        ax.set_ylabel('Second parameter value', fontsize=10)
        ax.set_zlabel('CVPE', fontsize=10)
        plt.title(CVPE_error_title)
        plt.savefig(CVPE_error_filename)

    return cvpe

