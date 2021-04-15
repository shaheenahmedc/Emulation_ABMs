def calculate_and_plot_cvpe_1d_manual_gpr(
                               used_parameter_settings, 
                               MSM_value_for_each_parameter_setting,
                               init_conds_length_scale_kernel_var_noise = [1.0,1.0,1.0],
                               bounds_length_scale_kernel_var_noise = (1e-5, None)
                               ):
    '''
    This function implements the cross-validation prediction error method of Barde and van der Hoog 2017. 
    It prints the GPR fit at each iteration, and the final differences between the GPR prediction at each parameter
    setting, and the actual value of the function at that parameter setting. 
    
    Inputs:
    used_parameter_settings = numpy array of used parameter settings.
    MSM_value_for_each_parameter_setting = MSM value for each parameter setting, averaged over repeated runs at each parameter setting. 
    init_conds_length_scale_kernel_var_noise = initial conditions for the optimisation of the length scale, kernel variance and noise parameters. 
    bounds_length_scale_kernel_var_noise = bounds for the values the three aforementioned parameters can take, during optimisation. 
    
    Outputs:
    cvpe = final CVPE value.
    '''
    cvpe_sum = 0
    Estimated_MSM_for_i_via_GPR_array = np.empty(len(used_parameter_settings))
    actual_MSM_for_i_array = np.empty(len(used_parameter_settings))
    for i in range(len(used_parameter_settings)):
        print (f'GPR CVPE loop is at {i}')
        # We delete i from the I-O arrays, and rerun GPR on this reduced set
        used_parameter_settings_without_i = np.delete(used_parameter_settings, i)
        MSM_value_for_each_parameter_setting_without_i = np.delete(MSM_value_for_each_parameter_setting, i)
        # We now run GPR on the I-O pairs, without i. But by inputting the input locations with i, we're able to extract
        # the estimate for GPR at i. Adjusting x changes the locations the GPR estimate is output for. 
        mu_vector_from_x_values = GPR_wrapper(
                x = used_parameter_settings.reshape(-1,1),
                x_train = used_parameter_settings_without_i.reshape(-1, 1),
                y_train = MSM_value_for_each_parameter_setting_without_i,
                init_conds_length_scale_kernel_var_noise = init_conds_length_scale_kernel_var_noise,
                bounds_length_scale_kernel_var_noise = bounds_length_scale_kernel_var_noise,
                naive = False, 
                method='L-BFGS-B')
        Estimated_MSM_for_i_via_GPR_array[i] = mu_vector_from_x_values[i] # For plotting cvpe 
        actual_MSM_for_i_array[i] = MSM_value_for_each_parameter_setting[i] # For plotting cvpe 
        Estimated_MSM_for_i_via_GPR = mu_vector_from_x_values[i] # Extract GPR estimate, on reduced I-O set, at i 
        actual_MSM_for_i = MSM_value_for_each_parameter_setting[i] # Compare to known evaluation of underlying function at i 
        squared_diff_between_actual_and_estimated_MSM = (actual_MSM_for_i - Estimated_MSM_for_i_via_GPR)**2
        cvpe_sum += squared_diff_between_actual_and_estimated_MSM

    cvpe = cvpe_sum/len(used_parameter_settings)

    plt.figure(figsize = (12,8))
    plt.plot(Estimated_MSM_for_i_via_GPR_array)
    plt.plot(actual_MSM_for_i_array)
    plt.legend()

    plt.figure(figsize = (12,8))
    plt.plot(Estimated_MSM_for_i_via_GPR_array - actual_MSM_for_i_array)
    plt.legend()
    plt.show()
    print (f'Final value for CVPE = {cvpe}')
    return cvpe

def calculate_and_plot_cvpe_1d_sklearn_gpr(
                                     used_parameter_settings,
                                     MSM_value_for_each_parameter_setting,
                                     init_conds_length_scale_kernel_var_noise = [1.0,1000,2]
                                     ):
    '''
    This function implements the cross-validation prediction error method of Barde and van der Hoog 2017, with sklearn.
    It prints the GPR fit at each iteration, and the final differences between the GPR prediction at each parameter
    setting, and the actual value of the function at that parameter setting. 
    It also outputs the final cvpe value. 
    
    Inputs:
    used_parameter_settings = numpy array of used parameter settings.
    MSM_value_for_each_parameter_setting = MSM value for each parameter setting, averaged over repeated runs at each parameter setting. 
    init_conds_length_scale_kernel_var_noise = initial conditions for the optimisation of the length scale, kernel variance and noise parameters. 
    
    Outputs:
    cvpe = final CVPE value.
    '''
    cvpe_sum = 0
    Estimated_MSM_for_i_via_GPR_array = np.empty(len(used_parameter_settings))
    actual_MSM_for_i_array = np.empty(len(used_parameter_settings))
    for i in range(len(used_parameter_settings)):
        print (f'GPR CVPE loop is at {i}')
        # We delete i from the I-O arrays, and rerun GPR on this reduced set
        used_parameter_settings_without_i = np.delete(used_parameter_settings, i)
        #print (f'parameter_settings_array = {parameter_settings_array}')
        #print (f'parameter_settings_array_without_i = {parameter_settings_array_without_i}')
        MSM_value_for_each_parameter_setting_without_i = np.delete(MSM_value_for_each_parameter_setting, i)
        # We now run GPR on the I-O pairs, without i. But by inputting the input locations with i, we're able to extract
        # the estimate for GPR at i. Adjusting x changes the locations the GPR estimate is output for. 
        mu_vector_from_x_values = GPR_wrapper_sklearn_sq_exp_kernel_plus_noise(used_parameter_settings, MSM_value_for_each_parameter_setting)
        Estimated_MSM_for_i_via_GPR_array[i] = mu_vector_from_x_values[i] # For plotting cvpe 
        actual_MSM_for_i_array[i] = MSM_value_for_each_parameter_setting[i] # For plotting cvpe 
        Estimated_MSM_for_i_via_GPR = mu_vector_from_x_values[i] # Extract GPR estimate, on reduced I-O set, at i 
        actual_MSM_for_i = MSM_value_for_each_parameter_setting[i] # Compare to known evaluation of underlying function at i 
        squared_diff_between_actual_and_estimated_MSM = (actual_MSM_for_i - Estimated_MSM_for_i_via_GPR)**2
        cvpe_sum += squared_diff_between_actual_and_estimated_MSM

    cvpe = cvpe_sum/len(used_parameter_settings)
    plt.figure(figsize = (12,8))
    plt.plot(Estimated_MSM_for_i_via_GPR_array)
    plt.plot(actual_MSM_for_i_array)
    plt.legend()

    plt.figure(figsize = (12,8))
    plt.plot(Estimated_MSM_for_i_via_GPR_array - actual_MSM_for_i_array)
    plt.legend()
    plt.show()
    print (f'Final value for CVPE, via sklearn = {cvpe}')
    return cvpe

def calculate_and_plot_cvpe_1d_gpy_gpr(used_parameter_settings,
                                     MSM_value_for_each_parameter_setting,
                                     init_conds_length_scale_kernel_var_noise = [1.0,1.0,1.0]
                                     ):
    '''
    This function implements the cross-validation prediction error method of Barde and van der Hoog 2017, with sklearn.
    It prints the GPR fit at each iteration, and the final differences between the GPR prediction at each parameter
    setting, and the actual value of the function at that parameter setting. 
    It also outputs the final cvpe value. 
    
    Inputs:
    used_parameter_settings = numpy array of used parameter settings.
    MSM_value_for_each_parameter_setting = MSM value for each parameter setting, averaged over repeated runs at each parameter setting. 
    init_conds_length_scale_kernel_var_noise = initial conditions for the optimisation of the length scale, kernel variance and noise parameters. 
    
    Outputs:
    cvpe = final CVPE value.
    '''
    cvpe_sum = 0
    Estimated_MSM_for_i_via_GPR_array = np.empty(len(used_parameter_settings))
    actual_MSM_for_i_array = np.empty(len(used_parameter_settings))
    for i in range(len(used_parameter_settings)):
        print (f'GPR CVPE loop is at {i}')
        # We delete i from the I-O arrays, and rerun GPR on this reduced set
        used_parameter_settings_without_i = np.delete(used_parameter_settings, i)
        #print (f'parameter_settings_array = {parameter_settings_array}')
        #print (f'parameter_settings_array_without_i = {parameter_settings_array_without_i}')
        MSM_value_for_each_parameter_setting_without_i = np.delete(MSM_value_for_each_parameter_setting, i)
        # We now run GPR on the I-O pairs, without i. But by inputting the input locations with i, we're able to extract
        # the estimate for GPR at i. Adjusting x changes the locations the GPR estimate is output for. 
        mu_vector_from_x_values = GPR_wrapper_gpy_sq_exp_kernel_plus_noise(used_parameter_settings,
                       MSM_value_for_each_parameter_setting,
                       init_conds_length_scale_kernel_var_noise)
        Estimated_MSM_for_i_via_GPR_array[i] = mu_vector_from_x_values[i] # For plotting cvpe 
        actual_MSM_for_i_array[i] = MSM_value_for_each_parameter_setting[i] # For plotting cvpe 
        Estimated_MSM_for_i_via_GPR = mu_vector_from_x_values[i] # Extract GPR estimate, on reduced I-O set, at i 
        actual_MSM_for_i = MSM_value_for_each_parameter_setting[i] # Compare to known evaluation of underlying function at i 
        squared_diff_between_actual_and_estimated_MSM = (actual_MSM_for_i - Estimated_MSM_for_i_via_GPR)**2
        cvpe_sum += squared_diff_between_actual_and_estimated_MSM

    cvpe = cvpe_sum/len(used_parameter_settings)
    plt.figure(figsize = (12,8))
    plt.plot(Estimated_MSM_for_i_via_GPR_array)
    plt.plot(actual_MSM_for_i_array)
    plt.legend()

    plt.figure(figsize = (12,8))
    plt.plot(Estimated_MSM_for_i_via_GPR_array - actual_MSM_for_i_array)
    plt.legend()
    plt.show()
    print (f'Final value for CVPE, via GPy = {cvpe}')
    return cvpe[0]
  
  def calculate_and_plot_param_settings_vs_MSM_fn_n_dim(
                                  parameter_lower_bounds_vector,
                                  parameter_upper_bounds_vector,
                                  vector_non_randomised_parameter_values,
                                  length_time_series,
                                  number_of_parameter_sets,
                                  number_of_repetitions_per_parameter_setting,
                                  true_parameter_value_vector,
                                  model_func,
                                  time_series_initial_values_vector = 0.0,
                                  number_of_moments = 7                            
                                  ):
    '''
    This function plots (where possible) parameter values against their distance function from a pseudo-true (or empirical) time series with known values.
    Now should work for n number of parameters.
    
    Inputs:
    parameter_lower_bounds_vector = vector of lower bounds for parameters.
    parameter_upper_bounds_vector = vector of upper bounds for parameters.
    length_time_series = length of pseudo-true/empirical time series. Model_generated series' will be some multiple of this (2 currently).
    number_of_parameter_sets = how many points in parameter range to sample. 
    number_of_repetitions_per_parameter_setting = how many repetitions at each parameter setting to undertake.
    true_parameter_value_vector = a vector of the parameter values used to generate the pseudo_true time series (set to None if empirical time series).
    model_func = a function which runs the model we're looking to calculate MSM for. Implemented in two locations below. For the pseudo true time series generation, and the random parameter set time series generation.
        It is hoped that such model_func's will only need two parameters: length_time_series and a params_vector. 
    time_series_initial_values_vector = If initial values are required for the model's time series output, pass them as an array here. 
    number_of_moments = how many moments to include in MSM calculation. 
    
    Outputs:
    used_parameter_settings = (d x n) numpy array of used parameter settings (d = dimensionality of parameter space, also number of rows. n = number of sample points in parameter space, also number of columns.)
    MSM_value_for_each_parameter_setting = Calculation of MSM for each parameter setting, averaged over repeated runs at each parameter setting. 
    '''
    if (len(parameter_lower_bounds_vector) != len(parameter_upper_bounds_vector)):
        print ('ERROR: BOUNDS OF DIFFERENT SHAPES')
    
    used_parameter_settings, indices_randomised_parameters = generate_random_points_in_n_dim_hypercube(len(parameter_lower_bounds_vector), number_of_parameter_sets, parameter_lower_bounds_vector, parameter_upper_bounds_vector, vector_non_randomised_parameter_values)
    pseudo_true_model_data = model_func(true_parameter_value_vector, length_time_series)
    MSM_value_for_each_parameter_setting = np.empty(number_of_parameter_sets)
    plt.figure()
    plt.plot(pseudo_true_model_data)
    plt.title('pseudo_true and model_generated data')
    for i in range(0, number_of_parameter_sets):
        print (f'loop number = {i}')
        random_parameter_setting = used_parameter_settings[:,i] # For MSM, we need to have the same params used each time, and same seed each time, so only result varies, stochasticity limited
        MSM_value_array_single_parameter_setting = np.empty(number_of_repetitions_per_parameter_setting)
        for j in range(0, number_of_repetitions_per_parameter_setting):
            seed(j*1234)
            model_generated_data = model_func(random_parameter_setting, length_time_series*2)
            plt.plot(model_generated_data)
            MSM_value = MSM_wrapper(pseudo_true_model_data, model_generated_data, number_of_moments, identity_matrix_bool = False) # Pull MSM_wrapper from Github download above. 
            MSM_value_array_single_parameter_setting[j] = MSM_value
        MSM_value_averaged = sum(MSM_value_array_single_parameter_setting)/number_of_repetitions_per_parameter_setting
        MSM_value_for_each_parameter_setting[i] = MSM_value_averaged
    
    if (len(indices_randomised_parameters) == 1):     
        plot_1_dim_MSM(used_parameter_settings[indices_randomised_parameters[0]], MSM_value_for_each_parameter_setting, true_parameter_value_vector[indices_randomised_parameters[0]])


    if (len(indices_randomised_parameters) == 2): 
        # MSM function can return nans if infinities hit, I believe only in BH model for certain param settings, but should account for this. 
        used_param_settings_with_sample_points_with_nan_MSM_removed, MSM_array_with_nans_removed = trim_nans_from_MSM_and_sample_points(used_parameter_settings, MSM_value_for_each_parameter_setting)
        first_dim_used_param_settings_with_MSM_nans_removed = used_param_settings_with_sample_points_with_nan_MSM_removed[indices_randomised_parameters[0]]
        second_dim_used_param_settings_with_MSM_nans_removed = used_param_settings_with_sample_points_with_nan_MSM_removed[indices_randomised_parameters[1]]
        
        plot_3d_MSM_figure(first_dim_used_param_settings_with_MSM_nans_removed, second_dim_used_param_settings_with_MSM_nans_removed, MSM_array_with_nans_removed)
    
    return used_parameter_settings, MSM_value_for_each_parameter_setting
  
def plot_3d_MSM_figure(first_dim_executed_param_values, secon_dim_executed_param_values, MSM_values):
    '''
    This function plots a 3d MSM surface, two varying parameters. 
    
    Inputs:
    first_dim_executed_param_values = (1 x (number of sample points - number of NANs in MSM)) shape numpy array, of first parameter to vary. 
    secon_dim_executed_param_values = (1 x (number of sample points - number of NANs in MSM)) shape numpy array, of second parameter to vary. 
    MSM_values = (number of sample points - number of NANs in MSM) shape numpy array, of non-NAN MSM values. 
    
    Output: 
    3d MSM plot.
    '''
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(first_dim_executed_param_values, secon_dim_executed_param_values, Z = MSM_values,cmap=cm.coolwarm)
    ax.set_xlabel('$X$', fontsize=20, rotation=150)
    ax.set_ylabel('$y$', fontsize=20, rotation=150)
