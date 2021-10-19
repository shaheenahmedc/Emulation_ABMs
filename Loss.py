import numpy as np
import matplotlib.pyplot as plt
import Thesis_modules.All_thesis_functions.Plotting as Plotting
import time
width = 386.67296
height = 624.25346

def transform_data_to_new_range(data, old_bounds, new_bounds):
    '''
    data = numpy array of data to transform from old_bounds to new_bounds
    old_bounds = numpy array, length 2, of old bounds
    new_bounds = numpy array, length 2, of new bounds
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

def loss_function(vec_theta_true, vec_theta_calibrated):
    abs_diff = np.abs(vec_theta_true - vec_theta_calibrated)
    return np.sqrt(np.sum(abs_diff**2)), abs_diff

def normalise_true_param_vector(parameter_vector, old_bounds, new_bounds):
    parameter_vector_normed = np.zeros(len(parameter_vector))

    for i in (range(0, len(parameter_vector))):
        parameter_vector_normed[i] = transform_data_to_new_range(parameter_vector[i],
                                                                        old_bounds[i],
                                                                        new_bounds)
    return parameter_vector_normed

def normalise_best_param_vector(parameter_vector, old_bounds, new_bounds, ALEBO_bool):
    parameter_vector_normed = np.zeros(len(parameter_vector))
    print (f'parameter_vector in normalise_best_param_vector = {parameter_vector}')
    for i in (range(0, len(parameter_vector))):
        if (ALEBO_bool):
            parameter_vector_normed[i] = transform_data_to_new_range(parameter_vector[i],
                                                                            old_bounds[i],
                                                                            new_bounds)
        else:
            parameter_vector_normed[i] = transform_data_to_new_range(parameter_vector[i],
                                                                old_bounds,
                                                                new_bounds)
    return parameter_vector_normed

def calculate_loss_and_produce_figure(best_sample_point_hist, 
                                      true_parameter_value_vector, 
                                      previous_BO_bounds, 
                                      original_param_bounds, 
                                      all_param_names_stripped_of_dollar,
                                      ALEBO_bool = False,
                                      filename = None):
    '''
    best_sample_point_hist = if ALEBO: shape (n_repeats x n_params). if BO: shape (n_trials x n_sequential_points x n_params)
    previous_BO_bounds = The bounds that the best_sample_points data came out in. E.g. in BOTorch it's [0,1], in BOTorch ALEBO it's [-1,1](?), but in AX ALEBO it's the original param set of the model.
    This means for ALEBO, we need to normalise each entry in the param vector by a different set of bounds. 
    original_param_bounds = Used for normalising the true_parameter_value_vector. 
    
    '''
    new_normalisation_bounds = [0,1]
    total_loss = 0
    if (ALEBO_bool):
        best_param_vector_at_end_of_each_trial = best_sample_point_hist
    else:
        best_param_vector_at_end_of_each_trial = best_sample_point_hist[:,-1,:]

    #best_param_vector_at_end_of_each_trial = best_sample_point_hist[:,-1,:]

    true_parameter_value_vector_normed = normalise_true_param_vector(true_parameter_value_vector, original_param_bounds, new_normalisation_bounds)
    total_loss_hist = np.zeros(best_param_vector_at_end_of_each_trial.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(Plotting.set_size(width)))
    for i in range(best_param_vector_at_end_of_each_trial.shape[0]):
        best_param_vector_at_end_of_one_trial_normed = normalise_best_param_vector(best_param_vector_at_end_of_each_trial[i],
                                                                              previous_BO_bounds,
                                                                              new_normalisation_bounds,
                                                                              ALEBO_bool) 

        summed_loss_for_trial, inidividual_losses = loss_function(true_parameter_value_vector_normed, best_param_vector_at_end_of_one_trial_normed)
        total_loss += summed_loss_for_trial
        total_loss_hist[i] = summed_loss_for_trial
        #indices = [i for i in range(0, len(inidividual_losses))]
        indices = all_param_names_stripped_of_dollar
        print (f'indices,inidividual_losses = {indices,inidividual_losses}')
        ax.scatter(indices,inidividual_losses, color = 'k', s = 2.5)
        for param_index in indices:
            plt.axvline(x=param_index, color = 'k', lw = 0.5)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #fig.suptitle(r'Losses - B')
    plt.xlabel('Parameter Name')
    plt.ylabel('Fractional Difference')
    plt.xticks(rotation=90)
    plt.savefig( filename + timestr + r'.pdf', format = 'pdf', bbox_inches='tight')

    loss_average = total_loss/len(total_loss_hist)
    loss_std_dev = np.std(total_loss_hist)
    return loss_average, loss_std_dev
