import Thesis_modules.All_thesis_functions.MSM as MSM
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from matplotlib import cm
import time
import Thesis_modules.All_thesis_functions.Plotting as Plotting
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import contextlib

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
def run_n_dim_frequentist_calibration(
                                  parameter_lower_bounds_vector,
                                  parameter_upper_bounds_vector,
                                  vector_non_randomised_parameter_values,
                                  length_time_series,
                                  number_of_parameter_sets,
                                  number_of_repetitions_per_parameter_setting,
                                  true_parameter_value_vector,
                                  model_func,
                                  fitness_func,
                                  data_print_filename,
                                  FC_figure_filename,
                                  data_print_title,
                                  FC_figure_title,
                                  equal_length_time_series_bool = False,
                                  number_of_moments = 7,
                                  sample_points_input = None,
                                  page_width = 386.67296,
                                  parameter_names = None,
                                  n_run = 1,
                                  use_median_fitness = False
                                  ):
    
    '''
    This function plots (where possible) parameter values against the distance function passed in, against a pseudo-true (or empirical) time series with known values.
    Now should work for n number of parameters.
    
    Parameters
    ----------
    parameter_lower_bounds_vector = vector of lower bounds for parameters.
    parameter_upper_bounds_vector = vector of upper bounds for parameters.
    length_time_series = length of pseudo-true/empirical time series. Model_generated series' will be some multiple of this (2 currently).
    number_of_parameter_sets = how many points in parameter range to sample. 
    number_of_repetitions_per_parameter_setting = how many repetitions at each parameter setting to undertake.
    true_parameter_value_vector = a vector of the parameter values used to generate the pseudo_true time series (set to None if empirical time series).
    model_func = a function which runs the model we're looking to calculate MSM for. Implemented in two locations below. For the pseudo true time series generation, and the random parameter set time series generation.
        It is hoped that such model_func's will only need two parameters: length_time_series and a params_vector. 
    fitness_func = a distance function to measure distance between pseudo-true/empirical time series and model generated time series. MSM/GSL/MIC etc. 
    data_print_filename = full file path for pseudo vs model generated figure. 
    FC_figure_filename = full file path for FC figure. 
    data_print_title = title for pseudo vs model generated figure. 
    FC_figure_title = title for FC figure. 
    equal_length_time_series_bool = If pseudo and model generated data should be of equal length (according to fitness function)
    number_of_moments = how many moments to include in MSM calculation. 
    sample_points_input = when we need to bypass random hypercube sampling, with an input of points (Bayesian Optimization)
    
    Outputs
    ----------
    used_parameter_settings = (d x n) numpy array of used parameter settings (d = dimensionality of parameter space, also number of rows. n = number of sample points in parameter space, also number of columns.)
    But if we input a set of sample points, currently we're inputting them as (n x d) and transposing
    MSM_value_for_each_parameter_setting = Calculation of MSM for each parameter setting, averaged over repeated runs at each parameter setting. 
    '''
    
    # Check bounds inputs same length
    # Add all other inputs which should be same length
    if (len(parameter_lower_bounds_vector) != len(parameter_upper_bounds_vector)):
        print ('ERROR: BOUNDS OF DIFFERENT SHAPES')
        
    # Check if we want to input a set of sample points (Bayesian Optimization)
    if (sample_points_input is None): # If not, run random hypercube sampling
            used_parameter_settings, indices_randomised_parameters = generate_random_points_in_n_dim_hypercube(len(parameter_lower_bounds_vector), number_of_parameter_sets, parameter_lower_bounds_vector, parameter_upper_bounds_vector, vector_non_randomised_parameter_values)
    else: # If so, transform the sample points input into the format required 
        #used_parameter_settings = sample_points_input[np.newaxis]
        used_parameter_settings = sample_points_input
        
        #used_parameter_settings = used_parameter_settings # Should transpose input from (samples x dim) to (dim x samples)

    # Sometimes, due to working between numpy arrays and pytorch tensors, extra dimensions enter the array.
    #used_parameter_settings = used_parameter_settings.squeeze()
    indices_randomised_parameters = np.where(vector_non_randomised_parameter_values == np.inf)[0]
    # Generate pseudo-true data
    # Pass an empirical_time_series? bool, to avoid having to re-write code for external calibration
    #seed(1234)
    #random.seed(1234)
    #pseudo_true_model_data = model_func(true_parameter_value_vector, length_time_series)

    # Initialise fitness values array
    fitness_value_for_each_parameter_setting = np.empty(number_of_parameter_sets)
    
    # Initialize figure for checking consistent values over multiple runs and same seed:
    fig, ax = plt.subplots()
        
        
    # Begin plotting - pseudo-true
    #plt.figure(figsize = Plotting.set_size(page_width))
    
    # Loop over sample points in parameter space, calculate distance to pseudo-true data
    for i in range(0, number_of_parameter_sets):

        # Get individual sample point 
        random_parameter_setting = used_parameter_settings[:,i] 
        
        # Initialise fitness array for storing repetitions. Different from final output fitness array.
        fitness_value_array_single_parameter_setting = np.empty(number_of_repetitions_per_parameter_setting)
        
        # Loop over repetitions for one sample point
        for j in range(0, number_of_repetitions_per_parameter_setting):
            parameter_names_copy_rndSeed = parameter_names.copy()
            parameter_names_copy_rndSeed.append('_rndSeed_')
            #seed((n_run*j+1)*1234)
            #with temp_seed(n_run*(j+1)*1234):
            with temp_seed(j*1234):
                pseudo_true_model_data = model_func(true_parameter_value_vector, 
                                                    length_time_series, 
                                                    parameter_names = parameter_names_copy_rndSeed,
                                                    seed_for_KS = j+1)
            # Seed should be different over repetitions
            # Seed affects performance massively! Numpy.random.seed or Python random.seed? 
            # Only use Numpy.random.seed, as Python random.seed seems to break MSM code? 
            #seed(int(time.time()))
            #seed(1234)
            #random.seed(1234)
            
            #with temp_seed(n_run*(j+1)*1234):
            with temp_seed(j*1234):
            # equal_length_time_series_bool tells us if we need to keep the two time series to be compared the same length (yes for GSL, no for MSM for instance)  
                if (equal_length_time_series_bool == True):                
                    model_generated_data = model_func(random_parameter_setting, 
                                                      length_time_series, 
                                                      parameter_names = parameter_names_copy_rndSeed,
                                                      seed_for_KS = j+1)
                else:
                    model_generated_data = model_func(random_parameter_setting, 
                                                      length_time_series*2, 
                                                      parameter_names = parameter_names_copy_rndSeed,
                                                      seed_for_KS = j+1)
            
            # Plot model generated data in red

            #plt.plot(model_generated_data, color = 'k', linewidth = 0.001)
            #plt.plot(pseudo_true_model_data, color = 'r', linewidth = 0.001)

            # Calculate and store repetitions of fitness between model generated and pseudo-true data, at one sample point
            fitness_value = fitness_func(pseudo_true_model_data, model_generated_data)        
            fitness_value_array_single_parameter_setting[j] = fitness_value          
        
            ax.scatter(random_parameter_setting[indices_randomised_parameters[0]],
                       fitness_value, color = 'r', s = 5, lw = 0.5, marker = 'x')
        # Average over repetitions 
        if (use_median_fitness):
            print ('Median fitness being used!')
            fitness_value_averaged = np.median(fitness_value_array_single_parameter_setting) # Use median fitness, if noise is very non-Gaussian (such as in KS)
            fitness_value_for_each_parameter_setting[i] = fitness_value_averaged

        else:
            fitness_value_averaged = sum(fitness_value_array_single_parameter_setting)/number_of_repetitions_per_parameter_setting
            fitness_value_for_each_parameter_setting[i] = fitness_value_averaged
        
        # Trim NaNs from fitness array, and relevant sample points
         
        #HASH OUT NAN TRIMS
        if (sample_points_input is None):
            used_param_settings_with_sample_points_with_nan_fitness_removed, fitness_array_with_nans_removed = trim_nans_from_fitness_and_sample_points(used_parameter_settings, fitness_value_for_each_parameter_setting)
        else:
            fitness_value_for_each_parameter_setting[np.isnan(fitness_value_for_each_parameter_setting)] = 1.0*10**3
            fitness_array_with_nans_removed = fitness_value_for_each_parameter_setting
            used_param_settings_with_sample_points_with_nan_fitness_removed = used_parameter_settings


    timestr = time.strftime("%Y%m%d-%H%M%S")
    #fig.savefig(r'C:\Users\shahe\OneDrive\Documents\Utrecht_19_20\Thesis\Numerical_Experiments\Figures\KS\mult_runs_test_' + timestr + '.pdf', format = 'pdf', bbox_inches = 'tight')

    # Plot pseudo-true data on top of model generated data
    #plt.plot(pseudo_true_model_data, color = 'k', label = 'Pseudo-True', linewidth = 0.5)
    #plt.title(data_print_title)
    #plt.legend(loc = 'upper right')
    plt.xlabel("Data point")
    plt.ylabel("Value")
    #plt.savefig(data_print_filename, format = 'pdf', bbox_inches='tight')
    plt.close()
    
    # Call relevant plotting function for fitness surface, based on dimensionality of sample point
    if (len(indices_randomised_parameters) == 1):     
        plot_1_dim_fitness_surface(used_param_settings_with_sample_points_with_nan_fitness_removed[indices_randomised_parameters[0]], 
                                   fitness_array_with_nans_removed, 
                                   true_parameter_value_vector[indices_randomised_parameters[0]], 
                                   FC_figure_filename, 
                                   FC_figure_title,
                                   page_width,
                                   parameter_names[indices_randomised_parameters[0]])
        
    if (len(indices_randomised_parameters) == 2): 
        if (number_of_parameter_sets == 1):
            pass
        else:
            first_dim_used_param_settings_with_fitness_nans_removed = used_param_settings_with_sample_points_with_nan_fitness_removed[indices_randomised_parameters[0]]
            second_dim_used_param_settings_with_fitness_nans_removed = used_param_settings_with_sample_points_with_nan_fitness_removed[indices_randomised_parameters[1]]
    
            plot_3d_fitness_surface(first_dim_used_param_settings_with_fitness_nans_removed, 
                                    second_dim_used_param_settings_with_fitness_nans_removed, 
                                    fitness_array_with_nans_removed, 
                                    FC_figure_filename, 
                                    FC_figure_title,
                                    page_width,
                                    true_parameter_value_vector,
                                    indices_randomised_parameters,
                                    parameter_names[indices_randomised_parameters[0]],
                                    parameter_names[indices_randomised_parameters[1]],
                                    parameter_lower_bounds_vector,
                                    parameter_upper_bounds_vector)
    
    # If we have input a set of sample points, and some/all of them come back NaN from the fitness calculation, 
    # Letham_20 can't handle it, so let's put in a very large value instead. Neaten this up later. 
    
    # Redundant?
    if (sample_points_input is not None):
        if (len(fitness_array_with_nans_removed) == 0):
            fitness_array_with_nans_removed = np.array([1.0*10**10])
            
    return used_param_settings_with_sample_points_with_nan_fitness_removed, fitness_array_with_nans_removed


def plot_1_dim_fitness_surface(used_parameter_settings, fitness_value_for_each_parameter_setting, true_parameter_value, FC_figure_filename, FC_figure_title, page_width, parameter_name):
    '''
    This function plots a 1d fitness surface, for one varying parameter.
    
    Parameters
    ----------
    used_parameter_settings = (1 x (number of sample points - number of NANs in fitness)) shape numpy array, of parameter to vary.  
    fitness_value_for_each_parameter_setting = (number of sample points - number of NANs in fitness) shape numpy array, of non-NAN fitness values. 
    
    Outputs
    ----------
    1d fitness plot. 
    '''
    print (f'true_parameter_value = {true_parameter_value}')
    '''
    plt.figure(figsize = Plotting.set_size(page_width))
    plt.scatter(used_parameter_settings, fitness_value_for_each_parameter_setting, s = 25, marker = '.', color = 'k')
    plt.xlabel(parameter_name)
    plt.ylabel("MSM value")
    if (true_parameter_value != None):
        plt.axvline(x = true_parameter_value, color = 'r')
    #plt.savefig(FC_figure_filename, format = 'pdf', bbox_inches='tight')
    #plt.close()
    '''

def plot_3d_fitness_surface(first_dim_executed_param_values, 
                            second_dim_executed_param_values, 
                            fitness_values, FC_figure_filename, 
                            FC_figure_title, 
                            page_width, 
                            true_parameter_vector, 
                            indices_randomised_parameters, 
                            first_param_name, 
                            second_param_name,
                            parameter_lower_bounds_vector,
                            parameter_upper_bounds_vector):
    '''
    This function plots a 3d fitness surface, two varying parameters. 
    
    Parameters
    ----------
    first_dim_executed_param_values = (1 x (number of sample points - number of NANs in fitness)) shape numpy array, of first parameter to vary. 
    secon_dim_executed_param_values = (1 x (number of sample points - number of NANs in fitness)) shape numpy array, of second parameter to vary. 
    fitness_values = (number of sample points - number of NANs in fitness) shape numpy array, of non-NAN fitness values. 
    
    Outputs
    ----------
    3d fitness plot.
    '''
    fig = plt.figure(figsize = Plotting.set_size(page_width))
    ax = fig.add_subplot(111)

    ax.set_xlabel(first_param_name, fontsize=10)
    ax.set_ylabel(second_param_name, fontsize=10)
    xi = np.linspace(min(first_dim_executed_param_values), max(first_dim_executed_param_values), 50)
    yi = np.linspace(min(second_dim_executed_param_values), max(second_dim_executed_param_values), 50)
    triang = tri.Triangulation(first_dim_executed_param_values, second_dim_executed_param_values)
    interpolator = tri.LinearTriInterpolator(triang, fitness_values)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    cntr1 = ax.contourf(xi, yi, zi, levels=50, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax)
    ax.set_xlim(parameter_lower_bounds_vector[indices_randomised_parameters[0]], parameter_upper_bounds_vector[indices_randomised_parameters[0]])
    ax.set_ylim(parameter_lower_bounds_vector[indices_randomised_parameters[1]], parameter_upper_bounds_vector[indices_randomised_parameters[1]])
    ax.scatter(true_parameter_vector[indices_randomised_parameters[0]], true_parameter_vector[indices_randomised_parameters[1]], color = 'r', marker = 'x')
    ax.scatter(first_dim_executed_param_values, second_dim_executed_param_values, color = 'k', marker = 'x')

    plt.savefig(FC_figure_filename, format = 'pdf', bbox_inches='tight')
    plt.close()


def trim_nans_from_fitness_and_sample_points(used_parameter_settings, fitness_value_for_each_parameter_setting): 
    '''
    This function removes NAN fitness values, and their corresponding parameter settings. 
    row in used_parameter_settings = parameter, column = sample point.
    
    Parameters
    ----------
    used_parameter_settings = (dimensionality x number of sample points) shape numpy array, of executed sample points in parameter space. 
    fitness_value_for_each_parameter_setting = (number of sample points) shape numpy array, of fitness values at each sample point. 
    
    Outputs
    ----------
    used_param_settings_with_sample_points_with_nan_fitness_removed = (d x (number of sample points - number of NANs in fitness)) shape numpy array. 
    fitness_array_with_nans_removed = (number of sample points - number of NANs in fitness) shape numpy array, of non-NAN fitness values. 
    '''
    #indices_of_varied_parameters = np.where(used_parameter_settings[:,0] == used_parameter_settings[:,1])[0] # The locations where first row of transposed (first sample point) = second sample point, should be indices of non-varied points in param space. 
    locations_of_nans_in_fitness =  np.where(np.isnan(fitness_value_for_each_parameter_setting))[0]

    used_param_settings_with_sample_points_with_nan_fitness_removed = np.delete(used_parameter_settings, locations_of_nans_in_fitness, axis = 1)
    fitness_array_with_nans_removed = np.delete(fitness_value_for_each_parameter_setting, locations_of_nans_in_fitness) 
    return used_param_settings_with_sample_points_with_nan_fitness_removed, fitness_array_with_nans_removed


def generate_random_points_in_n_dim_hypercube(number_of_dimensions, number_of_samples, lower_bounds_across_dims, upper_bounds_across_dims, vector_non_randomised_parameter_values): # Improvement: remove indices_randomised_parameters, can get from zeros of vector_non_randomised_parameter_values. 
    '''
    This function generates a desired number of random samples from an n-dimensional space.
    
    Parameters
    ----------
    number_of_dimensions = integer of space dimensionality.
    number_of_samples = integer of desired number of samples.
    lower_bounds_across_dims = numpy array of lower bounds for each dimension (in order). 
    upper_bounds_across_dims = numpy array of upper bounds for each dimension (in order). 
    vector_non_randomised_parameter_values = numpy array of constant values parameters not being randomised should take. Leave np.inf in place for randomised params to maintain indices.
    
    Outputs
    ----------
    sample_points = d x n numpy array of sample points (each column being a sample point).
    '''
    
    vector_non_randomised_parameter_values = vector_non_randomised_parameter_values.astype(float) # vector_non_randomised_parameter_values has to be floats, otherwise sample_points[i] change doesn't work. 
    indices_randomised_parameters = np.where(vector_non_randomised_parameter_values == np.inf)[0]
    sample_points = np.array([vector_non_randomised_parameter_values,]*number_of_samples).transpose()
    for i in range(0, number_of_dimensions):
        if (i in indices_randomised_parameters):
            sample_points[i] = np.random.uniform(low = lower_bounds_across_dims[i], high = upper_bounds_across_dims[i], size = number_of_samples) 
    return sample_points, indices_randomised_parameters

def calc_calibration_loss_function(vec_theta, vec_theta_g, lower_bounds, upper_bounds):
    total_length = len(vec_theta) + len(vec_theta_g) + len(lower_bounds) + len(upper_bounds)
    assert (total_length/(len(vec_theta)) ==  4) # Quick and dirty length assert
    vec_theta_normalised = (vec_theta - lower_bounds) / (upper_bounds - lower_bounds)
    vec_theta_g_normalised = (vec_theta_g - lower_bounds) / (upper_bounds - lower_bounds)
    loss_function = np.sum((vec_theta_normalised - vec_theta_g_normalised))**2
    loss_vector = (vec_theta_normalised - vec_theta_g_normalised)**2
    return loss_function
