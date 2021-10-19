# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:51:07 2021

@author: shahe
"""

import numpy as np

import time
import Thesis_modules.All_thesis_functions.Calibration as Calibration
import Thesis_modules.All_thesis_functions.Bayesian_Calibration as Bayesian_Calibration
import Thesis_modules.All_thesis_functions.MSM as MSM
import torch

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import standardize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement


def transform_data_to_new_range(data, old_bounds, new_bounds):
    '''
    Parameters
    ----------
    data = numpy array of data to transform from old_bounds to new_bounds
    old_bounds = numpy array, length 2, of old bounds
    new_bounds = numpy array, length 2, of new bounds
    
    Outputs
    ----------
    transformed_data = numpy array of transformed data

    '''

    old_min = old_bounds[0]
    old_max = old_bounds[1]
    new_min = new_bounds[0]
    new_max = new_bounds[1]
    old_value = data
    
    old_range = old_max - old_min
    new_range = new_max - new_min


    transformed_data = (((old_value - old_min) * new_range) / old_range) + new_min
    return transformed_data

def standardise_outputs(outputs_vector):
    mean_of_outputs = np.mean(outputs_vector.numpy())
    std_dev_of_outputs = np.std(outputs_vector.numpy())
    
    outputs_vector -= mean_of_outputs
    outputs_vector /= std_dev_of_outputs
    
    return outputs_vector, mean_of_outputs, std_dev_of_outputs

def obj(*,
       train_x_n_x_D,
       true_parameter_value_vector,
       BOTorch_bounds,
       parameter_lower_bounds_vector,
       parameter_upper_bounds_vector,
       array_bounds,
       vector_non_randomised_parameter_values,
       length_time_series,
       n_repeats_per_param_setting,
       model_func,
       parameter_names,
       Freq_or_Bayesian_bool, 
       pseudo_true_data = None,
       data_print_filename = None,
       FC_figure_filename = None,
       data_print_title = None,
       FC_figure_title = None):
    
    # Quick and dirty - check if Freq_or_Bayesian True and pseudo_true_data None, or vice versa:
    if ((Freq_or_Bayesian_bool) and (pseudo_true_data == None)):
        pass
    elif ((Freq_or_Bayesian_bool == False) and (pseudo_true_data is not None)):
        pass
    else:
        print ('ERROR - FREQ_OR_BAYESIAN_BOOL AND PSEUDO_TRUE_DATA DO NOT MATCH')
        
    # Set timestring for file names
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    # Get the indices of the parameters that are being varied
    indices_randomised_parameters = np.where(vector_non_randomised_parameter_values == np.inf)[0]

    # Transpose training data
    train_x_D_x_n = train_x_n_x_D.numpy().T
    
    # Create storage for reconstructed training data
    reconstructed_training_data = np.zeros((len(true_parameter_value_vector), train_x_D_x_n.shape[1]))
    
    # Transform training data 
    for i in range(0, train_x_D_x_n.shape[0]): # Loop over number of params in train_x
            train_x_D_x_n_transformed_row = transform_data_to_new_range(train_x_D_x_n[i,:], 
                                                                        BOTorch_bounds,
                                                                     array_bounds[indices_randomised_parameters[i]])
            reconstructed_training_data[indices_randomised_parameters[i],:] = train_x_D_x_n_transformed_row
    
    # Insert np.infs for params we'd like to vary 
    for j in range (0, len(true_parameter_value_vector)):
        if (vector_non_randomised_parameter_values[j] != np.inf):
            reconstructed_training_data[j,:] = vector_non_randomised_parameter_values[j]
            

    if (Freq_or_Bayesian_bool):
        
        # Extract number of sample points/parameter sets from training data
        number_of_parameter_sets = reconstructed_training_data.shape[1]
        
        # Execute run_n_dim_frequentist_calibration
        used_parameter_settings, fitness_value_for_each_parameter_setting = Calibration.run_n_dim_frequentist_calibration(
                                          parameter_lower_bounds_vector,
                                          parameter_upper_bounds_vector,
                                          vector_non_randomised_parameter_values,
                                          length_time_series,
                                          number_of_parameter_sets,
                                          n_repeats_per_param_setting,
                                          true_parameter_value_vector,
                                          model_func,
                                          MSM.MSM_wrapper,
                                          data_print_filename,
                                          FC_figure_filename,
                                          data_print_title,
                                          FC_figure_title,
                                          equal_length_time_series_bool = True,
                                          sample_points_input = reconstructed_training_data,
                                          page_width = 386.67296,
                                          parameter_names = parameter_names,
                                          use_median_fitness = True)
        
        return torch.tensor(fitness_value_for_each_parameter_setting)

    else: 
        fitnesses_storage = np.zeros(reconstructed_training_data.shape[1])
        # Run each training data point through BC function, get fitness       
        for i in range(0, reconstructed_training_data.shape[1]):
            fitness_value_for_each_parameter_setting, __ = Bayesian_Calibration.calc_posterior_of_sample_point(pseudo_true_data,
                                                                          reconstructed_training_data[:,i],
                                                                          true_parameter_value_vector,
                                                                          length_time_series,
                                                                          model_func,
                                                                          n_repeats_per_param_setting)
            fitnesses_storage[i] = fitness_value_for_each_parameter_setting

        return torch.tensor(fitnesses_storage)

    
    
def generate_initial_data(*,
                          bounds,
                           true_parameter_value_vector,
                           BOTorch_bounds,
                           parameter_lower_bounds_vector,
                           parameter_upper_bounds_vector,
                           array_bounds,
                           vector_non_randomised_parameter_values,
                           length_time_series,
                           n_repeats_per_param_setting,
                           model_func,
                           parameter_names,
                           Freq_or_Bayesian_bool, 
                           pseudo_true_data = None,
                           data_print_filename = None,
                           FC_figure_filename = None,
                           data_print_title = None,
                           FC_figure_title = None,
                            n=5):
    # generate training data
    # For n-dim models, it produces data in [0,1]^{n_varied_params}
    #train_x= draw_sobol_samples(bounds=bounds, n=n, q=1).item().squeeze(1)

    train_x = draw_sobol_samples(
        bounds=bounds, 
        n=n, 
        q=1, 
        seed=torch.randint(0,10000,(1,)).item()
        ).squeeze(1) # .squeeze removes unrequired middle dimension (n x 1 x D is output from draw_sobol_samples)
    
    exact_obj = obj(train_x_n_x_D = train_x,
               true_parameter_value_vector = true_parameter_value_vector,
               BOTorch_bounds = BOTorch_bounds,
               parameter_lower_bounds_vector = parameter_lower_bounds_vector,
               parameter_upper_bounds_vector = parameter_upper_bounds_vector,
               array_bounds = array_bounds,
               vector_non_randomised_parameter_values = vector_non_randomised_parameter_values,
               length_time_series = length_time_series,
               n_repeats_per_param_setting = n_repeats_per_param_setting,
               data_print_filename = data_print_filename,
               FC_figure_filename = FC_figure_filename,
               data_print_title = data_print_title,
               FC_figure_title = FC_figure_title,
               model_func = model_func,
               Freq_or_Bayesian_bool = Freq_or_Bayesian_bool,
               pseudo_true_data = pseudo_true_data,
               parameter_names = parameter_names).unsqueeze(-1)  # add output dimension


    # Return the training objectives from obj(), normalised by their mean and std. 
    # Also return the mean and std used to normalise them. 
    train_obj, mean_init_data, std_dev_init_data = standardise_outputs(exact_obj) 
    best_observed_value = train_obj.min().item()

    return train_x, train_obj, best_observed_value, mean_init_data, std_dev_init_data

def optimize_acqf_and_get_observation(*,
                          acq_func, 
                          bounds,
                           true_parameter_value_vector,
                           BOTorch_bounds,
                           parameter_lower_bounds_vector,
                           parameter_upper_bounds_vector,
                           array_bounds,
                           vector_non_randomised_parameter_values,
                           length_time_series,
                           n_repeats_per_param_setting,
                           model_func,
                           parameter_names,
                           Freq_or_Bayesian_bool, 
                           pseudo_true_data = None,
                           data_print_filename = None,
                           FC_figure_filename = None,
                           data_print_title = None,
                           FC_figure_title = None):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    
    num_restarts = 25
    raw_samples = 64
    
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    # observe new values 
    new_x = candidates.detach()

    # Candidate returned in [0,1]^D space. Input that to obj, and obj() converts to the bounds it wants.
    exact_obj = obj(train_x_n_x_D = new_x,
               true_parameter_value_vector = true_parameter_value_vector,
               BOTorch_bounds = BOTorch_bounds,
               parameter_lower_bounds_vector = parameter_lower_bounds_vector,
               parameter_upper_bounds_vector = parameter_upper_bounds_vector,
               array_bounds = array_bounds,
               vector_non_randomised_parameter_values = vector_non_randomised_parameter_values,
               length_time_series = length_time_series,
               n_repeats_per_param_setting = n_repeats_per_param_setting,
               data_print_filename = data_print_filename,
               FC_figure_filename = FC_figure_filename,
               data_print_title = data_print_title,
               FC_figure_title = FC_figure_title,
               model_func = model_func,
               Freq_or_Bayesian_bool = Freq_or_Bayesian_bool,
               pseudo_true_data = pseudo_true_data,
               parameter_names = parameter_names).unsqueeze(-1)  # add output dimension
    
    # training objective here un-normalised. Normalise in main BO loop (or here preferably)
    train_obj = exact_obj 
    return new_x, train_obj

def initialize_model(*, train_x, train_obj):
    #model = FixedNoiseGP(train_x, standardize(train_obj), train_yvar.expand_as(train_obj)).to(train_x) #.to(train_x)?
    model = SingleTaskGP(train_x, standardize(train_obj)) #.to(train_x)?

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model
    
def update_random_observations(*, 
                               best_random, 
                               mean_init_data, 
                               std_dev_init_data,
                               bounds,
                           true_parameter_value_vector,
                           BOTorch_bounds,
                           parameter_lower_bounds_vector,
                           parameter_upper_bounds_vector,
                           array_bounds,
                           vector_non_randomised_parameter_values,
                           length_time_series,
                           n_repeats_per_param_setting,
                           model_func,
                           parameter_names,
                           Freq_or_Bayesian_bool, 
                           pseudo_true_data = None,
                           data_print_filename = None,
                           FC_figure_filename = None,
                           data_print_title = None,
                           FC_figure_title = None):
    
    """Simulates a quasi-random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = draw_sobol_samples(bounds=bounds, n=1, q=1).squeeze(1)
    
    # draw_sobol_samples returns random sample points in [0,1]^D. obj() normalises to required dimensions.
    next_random_best = obj(train_x_n_x_D = rand_x,
               true_parameter_value_vector = true_parameter_value_vector,
               BOTorch_bounds = BOTorch_bounds,
               parameter_lower_bounds_vector = parameter_lower_bounds_vector,
               parameter_upper_bounds_vector = parameter_upper_bounds_vector,
               array_bounds = array_bounds,
               vector_non_randomised_parameter_values = vector_non_randomised_parameter_values,
               length_time_series = length_time_series,
               n_repeats_per_param_setting = n_repeats_per_param_setting,
               data_print_filename = data_print_filename,
               FC_figure_filename = FC_figure_filename,
               data_print_title = data_print_title,
               FC_figure_title = FC_figure_title,
               model_func = model_func,
               Freq_or_Bayesian_bool = Freq_or_Bayesian_bool,
               pseudo_true_data = pseudo_true_data,
               parameter_names = parameter_names).min().item()
    
    # Obj returns un-normalised objective. Normalise it
    next_random_best = (next_random_best - mean_init_data) / std_dev_init_data

    best_random.append(min(best_random[-1], next_random_best))  
     
    return best_random

def run_BO_loop(*,
                N_TRIALS = 2,
                N_BATCH = 3,
                n_init = 5,
                           true_parameter_value_vector,
                           BOTorch_bounds,
                           parameter_lower_bounds_vector,
                           parameter_upper_bounds_vector,
                           array_bounds,
                           vector_non_randomised_parameter_values,
                           length_time_series,
                           n_repeats_per_param_setting,
                           model_func,
                           parameter_names,
                           Freq_or_Bayesian_bool, 
                           pseudo_true_data = None,
                           data_print_filename = None,
                           FC_figure_filename = None,
                           data_print_title = None,
                           FC_figure_title = None):
    
    n_varied_params = len(np.where(vector_non_randomised_parameter_values == np.inf)[0])
    bounds = torch.tensor([[0.0] * n_varied_params, [1.0] * n_varied_params], device=device, dtype=dtype)

    best_observed_all_ei, best_random_all = [], []
    
    mean_init_data_vector = np.zeros(N_TRIALS)
    std_dev_init_data_vector = np.zeros(N_TRIALS)
    time_storage = np.zeros((N_TRIALS, N_BATCH))
    
    train_x_hist = np.zeros((N_TRIALS, N_BATCH+n_init, n_varied_params))
    best_sample_point_hist = np.zeros((N_TRIALS, N_BATCH+1, n_varied_params))
    
    for trial in range(1, N_TRIALS + 1):
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei,best_random = [], []
        
        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei, mean_init_data, std_dev_init_data = generate_initial_data(n=n_init,
               bounds = bounds,
               true_parameter_value_vector = true_parameter_value_vector,
               BOTorch_bounds = BOTorch_bounds,
               parameter_lower_bounds_vector = parameter_lower_bounds_vector,
               parameter_upper_bounds_vector = parameter_upper_bounds_vector,
               array_bounds = array_bounds,
               vector_non_randomised_parameter_values = vector_non_randomised_parameter_values,
               length_time_series = length_time_series,
               n_repeats_per_param_setting = n_repeats_per_param_setting,
               data_print_filename = data_print_filename,
               FC_figure_filename = FC_figure_filename,
               data_print_title = data_print_title,
               FC_figure_title = FC_figure_title,
               model_func = model_func,
               Freq_or_Bayesian_bool = Freq_or_Bayesian_bool,
               pseudo_true_data = pseudo_true_data,
               parameter_names = parameter_names)
        
        mean_init_data_vector[trial-1] = mean_init_data
        std_dev_init_data_vector[trial-1] = std_dev_init_data
        
        # Pass normalised training data, and normalised training objective

        print (f'train_x_ei = {train_x_ei}')
        print (f'train_obj_ei = {train_obj_ei}')
        mll_ei, model_ei = initialize_model(train_x = train_x_ei, 
                                            train_obj = train_obj_ei)      
        best_observed_ei.append(best_observed_value_ei)
        best_random.append(best_observed_value_ei)
        index_of_best_sample_point = train_obj_ei.argmin()
        best_sample_point_hist[trial-1,0,:] = train_x_ei[index_of_best_sample_point,:]
        
        for iteration in range(1, N_BATCH + 1):    
            print (f'Trial, iteration = {trial, iteration}')
            t0 = time.time()
        
            # fit the model
            fit_gpytorch_model(mll_ei)
            
            ei = ExpectedImprovement(
                model=model_ei, 
                best_f = min(train_obj_ei),
                maximize = False)
            
            '''
            # QNEI
            ei = qNoisyExpectedImprovement(
                model=model_ei, 
                X_baseline=train_x_ei)
            '''
            
            print ('fit + EI complete')
            
            # optimize and get new observation
            new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(
                acq_func = ei,
                bounds = bounds,
               true_parameter_value_vector = true_parameter_value_vector,
               BOTorch_bounds = BOTorch_bounds,
               parameter_lower_bounds_vector = parameter_lower_bounds_vector,
               parameter_upper_bounds_vector = parameter_upper_bounds_vector,
               array_bounds = array_bounds,
               vector_non_randomised_parameter_values = vector_non_randomised_parameter_values,
               length_time_series = length_time_series,
               n_repeats_per_param_setting = n_repeats_per_param_setting,
               data_print_filename = data_print_filename,
               FC_figure_filename = FC_figure_filename,
               data_print_title = data_print_title,
               FC_figure_title = FC_figure_title,
               model_func = model_func,
               Freq_or_Bayesian_bool = Freq_or_Bayesian_bool,
               pseudo_true_data = pseudo_true_data,
               parameter_names = parameter_names)
            
            new_obj_ei = (new_obj_ei - mean_init_data) / std_dev_init_data
            
            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
            
            # update progress
            best_random = update_random_observations(
                bounds = bounds,
                best_random = best_random, 
                mean_init_data = mean_init_data, 
                std_dev_init_data = std_dev_init_data,
                true_parameter_value_vector = true_parameter_value_vector,
               BOTorch_bounds = BOTorch_bounds,
               parameter_lower_bounds_vector = parameter_lower_bounds_vector,
               parameter_upper_bounds_vector = parameter_upper_bounds_vector,
               array_bounds = array_bounds,
               vector_non_randomised_parameter_values = vector_non_randomised_parameter_values,
               length_time_series = length_time_series,
               n_repeats_per_param_setting = n_repeats_per_param_setting,
               data_print_filename = data_print_filename,
               FC_figure_filename = FC_figure_filename,
               data_print_title = data_print_title,
               FC_figure_title = FC_figure_title,
               model_func = model_func,
               Freq_or_Bayesian_bool = Freq_or_Bayesian_bool,
               pseudo_true_data = pseudo_true_data,
               parameter_names = parameter_names)
            
            best_value_ei = train_obj_ei.min().item()
            index_of_best_sample_point = train_obj_ei.argmin()
            best_observed_ei.append(best_value_ei)
            best_sample_point_hist[trial-1,iteration,:] = train_x_ei[index_of_best_sample_point,:]
            mll_ei, model_ei = initialize_model(train_x = train_x_ei, 
                                            train_obj = train_obj_ei)  
            
            # Log times
            t1 = time.time()
            time_storage[trial-1, iteration-1] = (t1 - t0)
            
            print ('iteration complete')
        
        # Append data post iterations
        train_x_hist[trial-1,:,:] = train_x_ei
        best_observed_all_ei.append(best_observed_ei)
        best_random_all.append(best_random)
        print ('trial complete')
        
    best_observed_all_ei = np.asarray(best_observed_all_ei)
    best_random_all = np.asarray(best_random_all)
    for i in range(0, best_observed_all_ei.shape[0]):
        best_observed_all_ei[i] = (best_observed_all_ei[i]*std_dev_init_data_vector[i]) + mean_init_data_vector[i]
        best_random_all[i] = (best_random_all[i]*std_dev_init_data_vector[i]) + mean_init_data_vector[i]
        
    return best_observed_all_ei, best_random_all, best_sample_point_hist, train_obj_ei, train_x_ei, train_x_hist
        