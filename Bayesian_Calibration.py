# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:22:31 2021

@author: shahe
"""

from scipy import stats
import numpy as np
import Thesis_modules.All_thesis_functions.Calibration as Calibration
import Thesis_modules.All_thesis_functions.Plotting as Plotting
import Thesis_modules.All_thesis_functions.MSM as MSM
from numpy.random import seed
import matplotlib.pyplot as plt
import time
import contextlib
import numpy as np
from scipy.special import logsumexp

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
def BC_experiment(param_names,
                n_params, 
                n_sample_points,
                len_data,
                acceptance_tuning_param,
                plot_kde,
                n_repeats_per_param_setting,
                Uniform_value_lower_bound,
                lower_bounds_vector,
                upper_bounds_vector,
                vector_non_randomised_parameter_values,
                true_parameter_value_vector,
                model_func,
                width):
    
    #print (f'seed = {int(1000 * time.time()) % 2**32}')
    #seed(int(1000 * time.time()) % 2**32)
    # Generate random uniform numbers, for MCMC, here, because we set the seed below, and otherwise each number will be the same
    #O = np.random.uniform(Uniform_value_lower_bound, 1.0, size = n_sample_points*2)
    
    # Initialise eta vector, parameter vectors, parameter chain, posterior storage and pseudo-true data storage
    eta_vector = (upper_bounds_vector - lower_bounds_vector) / acceptance_tuning_param
    vec_theta_s = np.zeros(n_params)
    vec_theta_c = np.zeros(n_params)
    vec_theta_chain = np.zeros((n_sample_points, n_params))
    posterior_theta_s_hist = np.zeros(n_sample_points)

    # Generate random parameter set to initialise chain
    random_param_set = Calibration.generate_random_points_in_n_dim_hypercube(n_params, 
                                                          1, 
                                                          lower_bounds_vector, 
                                                          upper_bounds_vector, 
                                                          vector_non_randomised_parameter_values)[0]

    #random_param_set = true_parameter_value_vector*0.99
    # Set parameter vector to random parameter set
    vec_theta_s = random_param_set
    #vec_y = np.zeros(n_sample_points)
    
    
    '''
    Edit from Donovan feedback - hash in to restore previous version
    '''
    '''
    pseudo_true_data = np.zeros((len_data, n_repeats_per_param_setting))
    # Loop over number of repeats, for each param setting, and create pseudo-true data from multiple runs
    for i in range(0, n_repeats_per_param_setting):
        with temp_seed(i*1234):
            pseudo_true_data[:,i] = model_func(true_parameter_value_vector, len_data)
    '''
    '''
    Edit from Donovan feedback - hash in to restore previous version
    '''
    
    '''
    Edit from Donovan feedback - hash out to restore previous version
    '''
    
    n_repeats_of_simulated_data = 10
    #Instead of creating a pseudo-true matrix, over multiple runs, we need just one pseudo-true vector, at a random seed. 
    #Create new pseudo-true storage 
    pseudo_true_data = np.zeros(len_data)
    # Create single pseudo-true dataset
    #with temp_seed(1234):
    pseudo_true_data = model_func(true_parameter_value_vector, len_data)
    plt.plot(pseudo_true_data)
    plt.show()
    plt.close()
    '''
    Edit from Donovan feedback - hash out to restore previous version
    '''
    
    
    # Flatten pseudo-true data
    #pseudo_true_data = pseudo_true_data.flatten()

    # Create pseudo-true KDE PDF
    #kde_pseudo_true_data = stats.gaussian_kde(pseudo_true_data)

    # Initialise counters
    counter = 0
    acceptance_counter = 0
    l_theta_c_less_than_l_theta_s_counter = 0

    # Loop over number of sample points
    for i in range(n_sample_points):

        # Append theta_s to parameter chain
        vec_theta_chain[i, :] = vec_theta_s.flatten()
        # Copy theta_s into candidate parameter set
        vec_theta_c = vec_theta_s.copy()
        counter += 1
        print (f'counter = {counter}')

        # Loop over number of parameters, changing one of the ones we want to change
        for j in range(0, n_params):
            if (vector_non_randomised_parameter_values[j] != np.inf):
                continue
            else:
                pass
            # Keep trying to get a parameter set in the given bounds
            for k in range(0,100000):
                # Generate random variable from proposal distribution. 
                random_normal = np.random.normal(0.0, 1)
                # Generate new parameter value
                new_proposal = vec_theta_s[j] + eta_vector[j]*random_normal
                # Check in bounds, if valid parameter value, change parameter
                if (lower_bounds_vector[j] <= new_proposal <= upper_bounds_vector[j]):
                    vec_theta_c[j] = new_proposal
                    break # Stop search for new parameter value, move on to next parameter
                else:
                    continue # If new param value not in bounds, try again

        # Create scalar prior value
        bounds = upper_bounds_vector - lower_bounds_vector
        prior_scalar = np.prod(bounds)
        # Change prior to = 1, posterior proportional to likelihood x prior, and prior is constant
        prior_scalar = 1.0

        # Initialise theta_s storage for repeats of posterior calculation 
        posterior_for_theta_s_temp_storage = np.zeros(n_repeats_per_param_setting)
        # Loop over number of repeats per param setting
        
        theta_s_data_storage = np.zeros(len_data*n_repeats_of_simulated_data)
        for k in range(0, n_repeats_per_param_setting):
            
            '''
            Edit from Donovan feedback - hash out to restore previous version
            '''
            for m in range(0, n_repeats_of_simulated_data):
            # Generate data with theta_s
                with temp_seed(m*1234):
                    data_with_theta_s = model_func(vec_theta_s, len_data) 
            #Store data in array, over repeats
                theta_s_data_storage[m*len_data: (m+1)*len_data] = data_with_theta_s
                #print (f'vec_theta_s = {vec_theta_s}')
                #plt.plot(data_with_theta_s)
                #plt.show()
                #plt.close()
            #print (f'theta_s_data_storage = {theta_s_data_storage}')
            # Perform KDE on the entire theta_s dataset
            
            theta_s_data_storage = np.nan_to_num(theta_s_data_storage, nan=1.0*10**10, posinf=1.0*10**10, neginf=-1.0*10**10)
            kde_theta_s_data_storage = stats.gaussian_kde(theta_s_data_storage)
            # Evaluate the single previously generated pseudo-true dataset against the KDE of the repeated theta_s data
            kde_values_of_pseudo_true_data_against_theta_s = kde_theta_s_data_storage.evaluate(pseudo_true_data)
            kde_values_of_pseudo_true_data_against_theta_s = kde_values_of_pseudo_true_data_against_theta_s[kde_values_of_pseudo_true_data_against_theta_s != 0]

            # Take the sum(log) of the likelihoods of each datapoint, to avoid numerical underflow
            likelihood_of_pseudo_true_against_theta_s = np.sum(np.log(kde_values_of_pseudo_true_data_against_theta_s))
            print (f'likelihood_of_pseudo_true_against_theta_s = {likelihood_of_pseudo_true_against_theta_s}')

            '''
            Edit from Donovan feedback - hash out to restore previous version
            '''
            
            '''
            Edit from Donovan feedback - hash in to restore previous version
            '''
            '''
            # Form KDE distribution for pseudo-true data, for iteration k 
            kde_pseudo_true_data = stats.gaussian_kde(pseudo_true_data[:,k], bw_method = 'silverman')
            # Set seed, as the same seed used to generate this column of pseudo-true data
            with temp_seed(k*1234):
                # Generate data from model, with theta_s
                data_with_theta_s = model_func(vec_theta_s, len_data)
            # Calculate likelihood of theta_s 
            kde_values_data_with_theta_s = kde_pseudo_true_data.evaluate(data_with_theta_s)
            kde_values_data_with_theta_s = kde_values_data_with_theta_s[kde_values_data_with_theta_s != 0]
            #likelihood_of_data_with_theta_s = np.prod(kde_values_data_with_theta_s)
            #likelihood_of_data_with_theta_s = np.log(np.prod(kde_values_data_with_theta_s))
            #likelihood_of_data_with_theta_s = np.sum(np.log(kde_values_data_with_theta_s))
            likelihood_of_data_with_theta_s = -MSM.MSM_wrapper(pseudo_true_data[:,k], data_with_theta_s) + 1000
            '''
            '''
            Edit from Donovan feedback - hash in to restore previous version
            '''
            #print (f'number of zeros in KDE values = {np.count_nonzero(kde_values_data_with_theta_s==0)}')
            # Calculate posterior
            posterior_for_theta_s_iteration_k = likelihood_of_pseudo_true_against_theta_s * prior_scalar # uniform prior
            #Store posterior for this iteration 
            posterior_for_theta_s_temp_storage[k] = posterior_for_theta_s_iteration_k
        # Get mean of posteriors, over repeated runs at theta_s
        posterior_for_theta_s = np.mean(posterior_for_theta_s_temp_storage)
        # Store mean posterior
        posterior_theta_s_hist[i] = posterior_for_theta_s
        
        theta_c_data_storage = np.zeros(len_data*n_repeats_of_simulated_data)
        # Initialise theta_c storage for repeats of posterior calculation 
        posterior_for_theta_c_temp_storage = np.zeros(n_repeats_per_param_setting)
        # Loop over number of repeats per param setting
        for l in range(0, n_repeats_per_param_setting):
            
            '''
            Edit from Donovan feedback - hash out to restore previous version
            '''
            for m in range(0, n_repeats_of_simulated_data):
            # Generate data with theta_s
                with temp_seed(m*1234):
                # Generate data from model, with theta_c  
                    data_with_theta_c = model_func(vec_theta_c, len_data) 
            #Store data in array, over repeats
                theta_c_data_storage[m*len_data: (m+1)*len_data] = data_with_theta_c
            #print (f'theta_c_data_storage = {theta_c_data_storage}')
            
            theta_c_data_storage = np.nan_to_num(theta_c_data_storage, nan=1.0*10**10, posinf=1.0*10**10, neginf=-1.0*10**10)

            # Perform KDE on the entire theta_s dataset
            kde_theta_c_data_storage = stats.gaussian_kde(theta_c_data_storage, bw_method = 'silverman')
            # Evaluate the single previously generated pseudo-true dataset against the KDE of the repeated theta_s data
            kde_values_of_pseudo_true_data_against_theta_c = kde_theta_c_data_storage.evaluate(pseudo_true_data)
            # Take the sum(log) of the likelihoods of each datapoint, to avoid numerical underflow
            kde_values_of_pseudo_true_data_against_theta_c = kde_values_of_pseudo_true_data_against_theta_c[kde_values_of_pseudo_true_data_against_theta_c != 0]

            likelihood_of_pseudo_true_against_theta_c = np.sum(np.log(kde_values_of_pseudo_true_data_against_theta_c))
            if (likelihood_of_pseudo_true_against_theta_c == -np.inf):
                #print (f'theta_c_data_storage = {theta_c_data_storage}')
                print (f'kde_values_of_pseudo_true_data_against_theta_c = {kde_values_of_pseudo_true_data_against_theta_c}')
                print (len(kde_values_of_pseudo_true_data_against_theta_c))

            print (f'likelihood_of_pseudo_true_against_theta_c = {likelihood_of_pseudo_true_against_theta_c}')
            if (plot_kde == True):      
                points_for_kde = np.linspace(min(theta_c_data_storage), max(theta_c_data_storage), 1000)
                fig, ax = plt.subplots(figsize=Plotting.set_size(width))
                ax.hist(pseudo_true_data, density = True, bins=100, alpha=0.3)
                ax.plot(points_for_kde, kde_theta_c_data_storage(points_for_kde))
                plt.title(f'vec_theta_c = {vec_theta_c}')
                plt.show()
                plt.close()
            '''
            Edit from Donovan feedback - hash out to restore previous version
            '''
            
            '''
            Edit from Donovan feedback - hash in to restore previous version
            '''
            '''
            # Form KDE distribution for pseudo-true data, for iteration l 
            kde_pseudo_true_data = stats.gaussian_kde(pseudo_true_data[:,l], bw_method = 'silverman')
            # Set seed, as the same seed used to generate this column of pseudo-true data
            with temp_seed(l*1234):
                # Generate data from model, with theta_c        
                data_with_theta_c = model_func(vec_theta_c, len_data)
            # Choose whether to plot data generated with theta_c against KDE of theta_s or not
            if (plot_kde == True):      
                points_for_kde = np.linspace(min(data_with_theta_c), max(data_with_theta_c), 1000)
                fig, ax = plt.subplots(figsize=Plotting.set_size(width))
                ax.hist(data_with_theta_c, density = True, bins=100, alpha=0.3)
                ax.plot(points_for_kde, kde_pseudo_true_data(points_for_kde))
                plt.show()
                plt.close()
            # Calculate likelihood of theta_c
            kde_values_data_with_theta_c = kde_pseudo_true_data.evaluate(data_with_theta_c) 
            kde_values_data_with_theta_c = kde_values_data_with_theta_c[kde_values_data_with_theta_c != 0]

            #likelihood_of_data_with_theta_c = np.prod(kde_values_data_with_theta_c)
            #likelihood_of_data_with_theta_c = np.log(np.prod(kde_values_data_with_theta_c))
            #likelihood_of_data_with_theta_c = np.sum(np.log(kde_values_data_with_theta_c))
            likelihood_of_data_with_theta_c = - MSM.MSM_wrapper(pseudo_true_data[:,k], data_with_theta_c) + 1000
            '''
            '''
            Edit from Donovan feedback - hash in to restore previous version
            '''

            # Calculate posterior of theta_c 
            posterior_for_theta_c_iteration_l = likelihood_of_pseudo_true_against_theta_c * prior_scalar # uniform prior
            # Store posterior
            posterior_for_theta_c_temp_storage[l] = posterior_for_theta_c_iteration_l
        # Get mean of posteriors for theta_c, over repeated runs
        posterior_for_theta_c = np.mean(posterior_for_theta_c_temp_storage)
        # Calculate alpha 
        #ratio_of_likelihoods = posterior_for_theta_c/posterior_for_theta_s
        # Donovan's ratio - exp ratio 
        ratio_of_likelihoods = np.exp(posterior_for_theta_c - posterior_for_theta_s)
        if (ratio_of_likelihoods < 1):
            l_theta_c_less_than_l_theta_s_counter += 1
        alpha = min(1, ratio_of_likelihoods)
        
        # Generate random number, initialise seed
        #uniform_number = uniform_numbers_vector[i]
        uniform_number = np.random.uniform(Uniform_value_lower_bound, 1.0)

        # MCMC logic, select theta_c if better than theta_s always, but sometimes, even if worse, still select it. 
        if (alpha > uniform_number):
            vec_theta_s = vec_theta_c.copy()
            acceptance_counter += 1
        else:
            vec_theta_s = vec_theta_s.copy()
            
        print (f'alpha = {alpha}')
        print (f'posterior_for_theta_c = {posterior_for_theta_c}')
        print (f'posterior_for_theta_s = {posterior_for_theta_s}')


    print (f'acceptance_counter = {acceptance_counter}')
    print (f'l_theta_c_less_than_l_theta_s_counter = {l_theta_c_less_than_l_theta_s_counter}')

    # Get number of accepted points
    n_accepted_points = vec_theta_chain.shape[0]

    # Remove burn-in period (first 1/3 of chain)
    vec_theta_chain_burn_in_removed = vec_theta_chain[round(n_accepted_points/3.0):,:]   

    return vec_theta_chain, vec_theta_chain_burn_in_removed, posterior_theta_s_hist, acceptance_counter

def calc_posterior_of_sample_point(pseudo_true_data, param_vector, true_parameter_value_vector, len_data, model_func, n_repeats_per_param_setting, true_param_vector_bool = False, KS_bool = False, param_names_KS = None):    
    # Generate pseudo-empirical data
    n_repeats_of_simulated_data = 5
    #pseudo_true_data = np.zeros(len_data)
    #pseudo_true_data = model_func(true_parameter_value_vector, len_data)
    if (true_param_vector_bool):
        if (KS_bool):
            pseudo_true_data = model_func(true_parameter_value_vector, len_data, param_names_KS, 1)
        else:
            pseudo_true_data = model_func(true_parameter_value_vector, len_data)

    # Generate prior (= 1 for now)
    prior_scalar = 1.0
    counter = 0
    posterior_for_param_vector_temp_storage = np.zeros(n_repeats_per_param_setting)
    for k in range(0, n_repeats_per_param_setting):

        param_vector_data_storage = np.zeros(len_data*n_repeats_of_simulated_data)
        for m in range(0, n_repeats_of_simulated_data):
            counter +=1 
            # Generate data with theta_s
            with temp_seed(counter*1234):
                if (KS_bool):
                    parameter_names_copy_rndSeed = param_names_KS.copy()
                    parameter_names_copy_rndSeed.append('_rndSeed_')
                    print (f' seed = {(m+1) +(k*n_repeats_of_simulated_data)}')
                    data_with_param_vector = model_func(param_vector, len_data, parameter_names = parameter_names_copy_rndSeed, seed_for_KS = (m+1) +(k*n_repeats_of_simulated_data))
                else:
                    data_with_param_vector = model_func(param_vector, len_data)
                #Store data in array, over repeats
            param_vector_data_storage[m*len_data: (m+1)*len_data] = data_with_param_vector
        
        #if (np.isnan(param_vector_data_storage).any()):
            #print ('NANS IN MODEL DATA - CHECK NAN MAPPING VALUE')
        param_vector_data_storage = np.nan_to_num(param_vector_data_storage, nan=1.0*10**4, posinf=1.0*10**4, neginf=-1.0*10**4)
        kde_param_vector_data_storage = stats.gaussian_kde(param_vector_data_storage)
        # Evaluate the single previously generated pseudo-true dataset against the KDE of the repeated theta_s data
        kde_values_of_pseudo_true_data_against_param_vector = kde_param_vector_data_storage.evaluate(pseudo_true_data)
        kde_values_of_pseudo_true_data_against_param_vector = kde_values_of_pseudo_true_data_against_param_vector[kde_values_of_pseudo_true_data_against_param_vector != 0]
        
        # Take the sum(log) of the likelihoods of each datapoint, to avoid numerical underflow
        # Be aware, here we look at a different surface, the negative log-likelihood. 
        likelihood_of_pseudo_true_against_param_vector = -np.sum(np.log(kde_values_of_pseudo_true_data_against_param_vector))
        #likelihood_of_pseudo_true_against_param_vector = np.prod(kde_values_of_pseudo_true_data_against_param_vector)
        #print (likelihood_of_pseudo_true_against_param_vector)
        posterior_for_param_vector_iteration_k = likelihood_of_pseudo_true_against_param_vector * prior_scalar # uniform prior
        #Store posterior for this iteration 
        posterior_for_param_vector_temp_storage[k] = posterior_for_param_vector_iteration_k
        # Get mean of posteriors, over repeated runs at theta_s
        
    posterior_for_param_vector = np.mean(posterior_for_param_vector_temp_storage)
    std_dev_of_posterior = np.std(posterior_for_param_vector_temp_storage)
    #print (f'posterior_for_param_vector_temp_storage = {posterior_for_param_vector_temp_storage}')
    return posterior_for_param_vector, std_dev_of_posterior