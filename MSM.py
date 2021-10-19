import numpy as np 
import matplotlib.pyplot as plt
from numpy.random import seed
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
import Thesis_modules.All_thesis_functions.AR_1 as AR_1

def generate_moments_for_MSM(pseudo_true_data, model_generated_data):
    '''
    This function outputs the fractional difference between the seven designated moments, between a model_generated time series and pseudo_true (or empirical, but name hasn't been changed) time series.
    
    Parameters
    ---------- 
    pseudo_true_data = numpy array of a time series output from model with known parameters, or empirical time series. 
    model_generated_data = numpy array of a time series output from model, with test parameters.
    
    Outputs
    ----------
    fractional_difference_between_moments: numpy array, length 7, of fractional difference in moments, between model_generated and pseudo_true/empirical data.
    simple_weighting_matrix_diagonal_elements: diagonal elements of weighting matrix, as a vector. 
    '''
    
    '''
    Note - 10 Sep 2021:
    This code definitely can be simplified, as there's a lot of repeated calculations amongst the seven moments. 
    But it works currently, and doing so is very far down my priority list sadly. 
    If I end up adjusting the MSM calculation in the future, I will hope to refactor this code then. 
    '''

    pseudo_true_moments = np.empty(7)
    model_generated_moments = np.empty(7)

    '''
    Moment 1 - variance
    '''
    # Our longest lag = 5, therefore start at 4, loop from 0-4, 1-5, 2-6 etc
    longest_lag = 5
    m_i_model_generated_variance = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        m_i_z_t = np.std(z_t)
        m_i_model_generated_variance += m_i_z_t #This is our m_i(z_t^emp), want to store it, and calc inv sq diff to total 
    m_i_model_generated_variance = m_i_model_generated_variance/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[0] = m_i_model_generated_variance
      
    m_i_pseudo_true_variance = 0
    m_i_pseudo_true_variance_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t = np.std(z_t) # Calculate moment for given window of time series
        m_i_pseudo_true_variance += m_i_z_t # 
        m_i_pseudo_true_variance_history[i] = m_i_z_t # Append moment calc in mini window to storage array
    m_i_pseudo_true_variance = m_i_pseudo_true_variance/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[0] = m_i_pseudo_true_variance
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_variance_history_minus_avg_moment = m_i_pseudo_true_variance_history - m_i_pseudo_true_variance 
    m_i_pseudo_true_variance_history_minus_avg_moment_squared = m_i_pseudo_true_variance_history_minus_avg_moment**2
    w_i_j_1 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_variance_history_minus_avg_moment_squared))
    
    '''
    Moment 2 - Kurtosis
    '''
    m_i_model_generated_kurtosis = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        m_i_z_t = kurtosis(z_t)
        m_i_model_generated_kurtosis += m_i_z_t
    m_i_model_generated_kurtosis = m_i_model_generated_kurtosis/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[1] = m_i_model_generated_kurtosis
    
    m_i_pseudo_true_kurtosis = 0
    m_i_pseudo_true_kurtosis_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t = kurtosis(z_t)
        m_i_pseudo_true_kurtosis += m_i_z_t
        m_i_pseudo_true_kurtosis_history[i] = m_i_z_t
    m_i_pseudo_true_kurtosis = m_i_pseudo_true_kurtosis/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[1] = m_i_pseudo_true_kurtosis
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_kurtosis_history_minus_avg_moment = m_i_pseudo_true_kurtosis_history - m_i_pseudo_true_kurtosis 
    m_i_pseudo_true_kurtosis_history_minus_avg_moment_squared = m_i_pseudo_true_kurtosis_history_minus_avg_moment**2
    w_i_j_2 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_kurtosis_history_minus_avg_moment_squared))

    '''
    Moment 3 - Autocorrelation function at lag 1
    '''
    m_i_model_generated_acf_lag_1 = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        m_i_model_generated_acf_lag_1 += acf(z_t, nlags = 1)[1]
    m_i_model_generated_acf_lag_1 = m_i_model_generated_acf_lag_1/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[2] = m_i_model_generated_acf_lag_1
  
    m_i_pseudo_true_acf_lag_1 = 0
    m_i_pseudo_true_acf_lag_1_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t = acf(z_t, nlags = 1)[1]
        m_i_pseudo_true_acf_lag_1 += m_i_z_t
        m_i_pseudo_true_acf_lag_1_history[i] = m_i_z_t
    m_i_pseudo_true_acf_lag_1 = m_i_pseudo_true_acf_lag_1/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[2] = m_i_pseudo_true_acf_lag_1
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_acf_lag_1_history_minus_avg_moment = m_i_pseudo_true_acf_lag_1_history - m_i_pseudo_true_acf_lag_1 
    m_i_pseudo_true_acf_lag_1_history_minus_avg_moment_squared = m_i_pseudo_true_acf_lag_1_history_minus_avg_moment**2
    w_i_j_3 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_acf_lag_1_history_minus_avg_moment_squared))
    
    
    '''
    Moment 4 - Autocorrelation function of squared series at lag 1
    '''
    m_i_model_generated_acf_squared_lag_1 = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        m_i_model_generated_acf_squared_lag_1 += acf(z_t**2, nlags = 1)[1]
    m_i_model_generated_acf_squared_lag_1 = m_i_model_generated_acf_squared_lag_1/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[3] = m_i_model_generated_acf_squared_lag_1

    m_i_pseudo_true_acf_squared_lag_1 = 0
    m_i_pseudo_true_acf_squared_lag_1_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t = acf(z_t**2, nlags = 1)[1]
        m_i_pseudo_true_acf_squared_lag_1 += m_i_z_t
        m_i_pseudo_true_acf_squared_lag_1_history[i] = m_i_z_t
    m_i_pseudo_true_acf_squared_lag_1 = m_i_pseudo_true_acf_squared_lag_1/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[3] = m_i_pseudo_true_acf_squared_lag_1
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_acf_squared_lag_1_history_minus_avg_moment = m_i_pseudo_true_acf_squared_lag_1_history - m_i_pseudo_true_acf_squared_lag_1 
    m_i_pseudo_true_acf_squared_lag_1_history_minus_avg_moment_squared = m_i_pseudo_true_acf_squared_lag_1_history_minus_avg_moment**2
    w_i_j_4 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_acf_squared_lag_1_history_minus_avg_moment_squared))
    
    
    '''
    Moment 5 - Autocorrelation function of absolute series at lag 1
    '''
    m_i_model_generated_acf_abs_lag_1 = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        m_i_model_generated_acf_abs_lag_1 += acf(np.abs(z_t), nlags = 1)[1]
    m_i_model_generated_acf_abs_lag_1 = m_i_model_generated_acf_abs_lag_1/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[4] = m_i_model_generated_acf_abs_lag_1

    m_i_pseudo_true_acf_abs_lag_1 = 0
    m_i_pseudo_true_acf_abs_lag_1_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t = acf(np.abs(z_t), nlags = 1)[1]
        m_i_pseudo_true_acf_abs_lag_1 += m_i_z_t
        m_i_pseudo_true_acf_abs_lag_1_history[i] = m_i_z_t
    m_i_pseudo_true_acf_abs_lag_1 = m_i_pseudo_true_acf_abs_lag_1/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[4] = m_i_pseudo_true_acf_abs_lag_1
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_acf_abs_lag_1_history_minus_avg_moment = m_i_pseudo_true_acf_abs_lag_1_history - m_i_pseudo_true_acf_abs_lag_1 
    m_i_pseudo_true_acf_abs_lag_1_history_minus_avg_moment_squared = m_i_pseudo_true_acf_abs_lag_1_history_minus_avg_moment**2
    w_i_j_5 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_acf_abs_lag_1_history_minus_avg_moment_squared))
    
    
    '''
    Moment 6 - Autocorrelation function of squared series at lag 5
    '''
    m_i_model_generated_acf_squared_lag_5 = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        #print (f'z_t = {z_t}')
        #print (f'acf = {acf(z_t**2, nlags = 5)}')
        m_i_model_generated_acf_squared_lag_5 += acf(z_t**2, nlags = 5)[5]
    m_i_model_generated_acf_squared_lag_5 = m_i_model_generated_acf_squared_lag_5/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[5] = m_i_model_generated_acf_squared_lag_5

    m_i_pseudo_true_acf_squared_lag_5 = 0
    m_i_pseudo_true_acf_squared_lag_5_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t = acf(z_t**2, nlags = 5)[5]
        m_i_pseudo_true_acf_squared_lag_5 += m_i_z_t
        m_i_pseudo_true_acf_squared_lag_5_history[i] = m_i_z_t
    m_i_pseudo_true_acf_squared_lag_5 = m_i_pseudo_true_acf_squared_lag_5/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[5] = m_i_pseudo_true_acf_squared_lag_5
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_acf_squared_lag_5_history_minus_avg_moment = m_i_pseudo_true_acf_squared_lag_5_history - m_i_pseudo_true_acf_squared_lag_5 
    m_i_pseudo_true_acf_squared_lag_5_history_minus_avg_moment_squared = m_i_pseudo_true_acf_squared_lag_5_history_minus_avg_moment**2
    w_i_j_6 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_acf_squared_lag_5_history_minus_avg_moment_squared))
    
    '''
    Moment 7 - Autocorrelation function of absolute series at lag 5
    '''
    m_i_model_generated_acf_abs_lag_5 = 0
    for i in range(0, len(model_generated_data)-(longest_lag + 1)):
        z_t = model_generated_data[i:i+(longest_lag + 1)]
        m_i_model_generated_acf_abs_lag_5 += acf(np.abs(z_t), nlags = 5)[5]
    m_i_model_generated_acf_abs_lag_5 = m_i_model_generated_acf_abs_lag_5/len(model_generated_data) #We haven't performed T additions, with starting loop at 4   
    model_generated_moments[6] = m_i_model_generated_acf_abs_lag_5
    
    m_i_pseudo_true_acf_abs_lag_5 = 0
    m_i_pseudo_true_acf_abs_lag_5_history = np.empty(len(pseudo_true_data) - (longest_lag + 1)) # Initialise storage of rolling moments calculation
    for i in range(0, len(pseudo_true_data)-(longest_lag + 1)): # Sum in equation 1 of Franke_09 starts at 1?
        z_t = pseudo_true_data[i:i+(longest_lag + 1)]
        m_i_z_t =  acf(np.abs(z_t), nlags = 5)[5]
        m_i_pseudo_true_acf_abs_lag_5 += m_i_z_t
        m_i_pseudo_true_acf_abs_lag_5_history[i] = m_i_z_t
    m_i_pseudo_true_acf_abs_lag_5 = m_i_pseudo_true_acf_abs_lag_5/len(pseudo_true_data) #We haven't performed T additions, with starting loop at 4   
    pseudo_true_moments[6] = m_i_pseudo_true_acf_abs_lag_5
    # Calculate entry in simple diagonal weighting matrix - Franke_09
    # Debate in literature if subtracting pseudo_true/empirical or model_data best
    m_i_pseudo_true_acf_abs_lag_5_history_minus_avg_moment = m_i_pseudo_true_acf_abs_lag_5_history - m_i_pseudo_true_acf_abs_lag_5
    m_i_pseudo_true_acf_abs_lag_5_history_minus_avg_moment_squared = m_i_pseudo_true_acf_abs_lag_5_history_minus_avg_moment**2
    w_i_j_7 = 1.0 / ((1.0/len(pseudo_true_data)) * np.sum(m_i_pseudo_true_acf_abs_lag_5_history_minus_avg_moment_squared))

    #Combine weighting matrix elements, and calculate frac_diff
    simple_weighting_matrix_diagonal_elements = np.array([w_i_j_1, w_i_j_2, w_i_j_3, w_i_j_4, w_i_j_5, w_i_j_6, w_i_j_7])
    fractional_difference_between_moments = (model_generated_moments - pseudo_true_moments) / pseudo_true_moments
    return fractional_difference_between_moments, simple_weighting_matrix_diagonal_elements

def apply_weighting_matrix_to_moments(fractional_difference_between_moments, simple_weighting_matrix_diagonal_elements, number_of_moments = 7,  identity_matrix_bool = False):
    '''
    This function applies matrix W to fractional moments difference vector.
    Weights for uncertainty in each moment.
    Can set to identity matrix initially. 
    
    Parameters
    ---------- 
    fractional_difference_between_moments:vector of seven fractional differences in moments, between model_generated and pseudo_true or empirical data.
    number_of_moments: how many of the moments to include in the MSM calculation. Trims matrices to exclude.
    identity_matrix_bool: boolean for using identity matrix or not.
    simple_weighting_matrix_diagonal_elements: vector of diagonal elements. Makes bool before obsolete, fix.
    
    Outputs
    ---------- 
    frac_diff_transposed_dot_s_dot_frac_diff: numpy array, length 7, of fractional difference in moments, between model_generated and pseudo_true_or_empirical data, weighted for uncertainty.
    '''
    
    if (identity_matrix_bool == True):   
        identity_matrix = np.identity(number_of_moments, dtype = float) 
        i_dot_frac_diff = identity_matrix.dot(fractional_difference_between_moments[0:number_of_moments])
        frac_diff_transposed = fractional_difference_between_moments[0:number_of_moments].T
        frac_diff_transposed_dot_i_dot_frac_diff = frac_diff_transposed.dot(i_dot_frac_diff)
        return frac_diff_transposed_dot_i_dot_frac_diff
    else:
        simple_weighting_matrix = np.identity(number_of_moments, dtype = float) 
        np.fill_diagonal(simple_weighting_matrix, simple_weighting_matrix_diagonal_elements)
        s_dot_frac_diff = simple_weighting_matrix.dot(fractional_difference_between_moments[0:number_of_moments])
        frac_diff_transposed = fractional_difference_between_moments[0:number_of_moments].T
        frac_diff_transposed_dot_s_dot_frac_diff = frac_diff_transposed.dot(s_dot_frac_diff)
        return frac_diff_transposed_dot_s_dot_frac_diff        

    
def MSM_wrapper(pseudo_true_or_empirical_data, model_generated_data):
    '''
    This function wraps the fractional moment difference generator and the weighting matrix application into one function.
    
    Parameters
    ---------- 
    pseudo_true_or_empirical_data: numpy array of either a time series output from model with known parameters, or real data.
    model_generated_data: numpy array of a time series output from model, with test parameters.
    number_of_moments: how many of the moments to include in the MSM calculation. Trims matrices to exclude.
    identity_matrix_bool: boolean for using identity matrix or not.
    
    Outputs
    ---------- 
    distance_measure: Scalar distance measure between two input time series
    '''
    frac_diff_moments, simple_weighting_matrix_diagonal_elements = generate_moments_for_MSM(pseudo_true_or_empirical_data, model_generated_data)
    distance_measure = apply_weighting_matrix_to_moments(frac_diff_moments, simple_weighting_matrix_diagonal_elements = simple_weighting_matrix_diagonal_elements)
    return distance_measure

def calculate_and_plot_param_settings_vs_MSM_fn_1d(
                                  lower_bound_for_parameter,
                                  upper_bound_for_parameter,
                                  length_time_series,
                                  number_of_parameter_sets,
                                  number_of_repetitions_per_parameter_setting,
                                  true_parameter_value,
                                  initial_value = 0.0,
                                  number_of_moments = 7
                                  ):
    '''
    This function plots parameter values against their distance function from a pseudo-true time series with known values.
    Only work for models with one parameter.
    
    Parameters
    ---------- 
    lower_bound_for_parameter: lower bound for pseudo-true time series.
    upper_bound_for_parameter: upper bound for pseudo-true time series.
    length_time_series: length of pseudo-true time series. Model_generated series' will be some multiple of this (2 currently).
    number_of_parameter_sets: how many points in parameter range to sample. 
    number_of_repetitions_per_parameter_setting: how many repetitions at each parameter setting to undertake.
    true_parameter_value: the parameter value used to generate the pseudo-true time series.
    initial_value: In the AR(1) model, an initial value for the time series needs to be provided.
    number_of_moments: how many moments to include in MSM calculation. 
    
    Outputs
    ---------- 
    used_parameter_settings: numpy array of used parameter settings
    MSM_value_for_each_parameter_setting: Calculation of MSM for each parameter setting, averaged over repeated runs at each parameter setting. 
    '''
    used_parameter_settings = np.linspace(lower_bound_for_parameter, upper_bound_for_parameter, number_of_parameter_sets)
    pseudo_true_ar1_data = generate_ar1_series(true_parameter_value, length_time_series)
    MSM_value_for_each_parameter_setting = np.empty(number_of_parameter_sets)

    for i in range(number_of_parameter_sets):
        print (f'loop number = {i}')
        # Note, we should implement the ability to replicate at given parameter values, not just take more 
        random_parameter_setting = used_parameter_settings[i] # For MSM, we need to have the same params used each time, and same seed each time, so only result varies, stochasticity limited
        MSM_value_array = np.empty(number_of_repetitions_per_parameter_setting)
        for j in range(0, number_of_repetitions_per_parameter_setting):
            seed(j*1234)
            model_generated_data = generate_ar1_series(random_parameter_setting, length_time_series*2)
            #plot_ar1_series(model_generated_data, random_parameter_setting)
            MSM_value = MSM_wrapper(pseudo_true_ar1_data, model_generated_data, number_of_moments, identity_matrix_bool = False)
            MSM_value_array[j] = MSM_value
        MSM_value_averaged = sum(MSM_value_array)/number_of_repetitions_per_parameter_setting
        MSM_value_for_each_parameter_setting[i] = MSM_value_averaged
    
    plt.figure()
    plt.scatter(used_parameter_settings, MSM_value_for_each_parameter_setting, s = 2)
    plt.xlabel("Parameter value")
    plt.ylabel("MSM value")
    plt.axvline(x = true_parameter_value, color = 'r')
    plt.show()
    
    return used_parameter_settings, MSM_value_for_each_parameter_setting

def calc_and_plot_MSM_empirical_vs_model_generated(
                                  lower_bound_for_parameter,
                                  upper_bound_for_parameter,
                                  empirical_time_series,
                                  number_of_parameter_sets,
                                  number_of_repetitions_per_parameter_setting,
                                  initial_value = 0.0
                                  ):
    '''
    This function plots parameter values against their distance function from an empirical time series with known values.
    Only work for models with one parameter.
    
    Parameters
    ---------- 
    lower_bound_for_parameter = lower bound for best estimate of parameter in empirical time series.
    upper_bound_for_parameter = upper bound for best estimate of parameter in empirical time series.
    empirical_time_series = numpy array of empirical time series.
    number_of_parameter_sets = how many points in parameter range to sample. 
    number_of_repetitions_per_parameter_setting = how many repetitions at each parameter setting to undertake.
    initial_value = In the AR(1) model, an initial value for the time series needs to be provided.
    
    Outputs
    ---------- 
    used_parameter_settings = numpy array of used parameter settings
    MSM_value_for_each_parameter_setting = Calculation of MSM for each parameter setting, averaged over repeated runs at each parameter setting. 
    '''
    
    used_parameter_settings = np.linspace(lower_bound_for_parameter, upper_bound_for_parameter, number_of_parameter_sets)
    MSM_value_for_each_parameter_setting = np.empty(number_of_parameter_sets)
    length_time_series = len(empirical_time_series)
    for i in range(number_of_parameter_sets):
        print (f'loop number = {i}')
        # Note, we should implement the ability to replicate at given parameter values, not just take more 
        random_parameter_setting = np.array([used_parameter_settings[i]]) # For MSM, we need to have the same params used each time, and same seed each time, so only result varies, stochasticity limited
        MSM_value_array = np.empty(number_of_repetitions_per_parameter_setting)
        for j in range(0, number_of_repetitions_per_parameter_setting):
            seed(j*1234)
            model_generated_data = AR_1.generate_ar1_series(random_parameter_setting, length_time_series*2)
            MSM_value = MSM_wrapper(empirical_time_series, model_generated_data)
            MSM_value_array[j] = MSM_value
        MSM_value_averaged = sum(MSM_value_array)/number_of_repetitions_per_parameter_setting
        MSM_value_for_each_parameter_setting[i] = MSM_value_averaged
    
    plt.figure()
    plt.scatter(used_parameter_settings, MSM_value_for_each_parameter_setting, s = 2)
    plt.xlabel("Parameter value")
    plt.ylabel("MSM value")
    plt.show()
    
    return used_parameter_settings, MSM_value_for_each_parameter_setting


def plot_1_dim_MSM(used_parameter_settings, MSM_value_for_each_parameter_setting, true_parameter_value):
    '''
    This function plots a 1d MSM surface, for one varying parameter.
    
    Parameters
    ---------- 
    used_parameter_settings = (1 x (number of sample points - number of NANs in MSM)) shape numpy array, of parameter to vary.  
    MSM_values = (number of sample points - number of NANs in MSM) shape numpy array, of non-NAN MSM values. 
    
    Outputs
    ---------- 
    1d MSM plot. 
    '''
    
    plt.figure()
    plt.scatter(used_parameter_settings, MSM_value_for_each_parameter_setting, s = 2)
    plt.xlabel("Parameter value")
    plt.ylabel("MSM value")
    if (true_parameter_value != None):
        plt.axvline(x = true_parameter_value, color = 'r')
    plt.show()
