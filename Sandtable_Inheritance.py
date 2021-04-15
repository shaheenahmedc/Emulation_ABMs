import subprocess
import yaml
import pandas as pd 
import numpy as np
import os

def change_parameter_in_yaml(full_filepath, parameter, new_value):
    with open(full_filepath) as f:
        doc = yaml.safe_load(f)
        for d in doc['model_parameters']:
            if isinstance(d, dict):
                if d.get('name') == parameter:
                    d['value'] = new_value
                
    with open(full_filepath, 'w') as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys = False)

def run_SA_on_Sandtable_Inheritance(number_of_dimensions,
                                   number_of_samples,
                                   lower_bounds_across_dims, 
                                   upper_bounds_across_dims,
                                       vector_non_randomised_parameter_values):
    '''
    This function runs a sensitivity analysis over a given set of parameters in the Sandtable Inheritance model. 
    
    Inputs:
    number_of_dimensions = integer value of the number of varying parameters/dimensionality of the parameter space to sample from.  
    number_of_samples = number of sample points in the parameter space to test. 
    lower_bounds_across_dims = numpy array of lower bounds for each dimension (in order). 
    upper_bounds_across_dims = numpy array of upper bounds for each dimension (in order).
    vector_non_randomised_parameter_values = numpy array of constant values parameters not being randomised should take. Leave np.inf in place for randomised params to maintain indices.
    Output: 
    used_parameter_settings = (dimensionality x number_of_samples) shape numpy array, of sampled points. Sampling done randomly, not via Latin Hypercube. 
    fitness_function_of_gini = a subset of the Gini values to take as the fitness function. 
    
    Note: Generalise this function for any model, as done for calculate_and_plot_param_settings_vs_MSM_fn_n_dim. 
    Allow any sampling function, which will return us a set of points within a hypercube, as well as any model function. 
    '''    
    used_parameter_settings, _ = MSM.generate_random_points_in_n_dim_hypercube(number_of_dimensions, number_of_samples, lower_bounds_across_dims, upper_bounds_across_dims, vector_non_randomised_parameter_values)
    sample_point_ticker = 0
    gini_values_at_year_50 = np.empty(number_of_samples)
    for sample_point in used_parameter_settings.T:
        parameter_names = ['assortative_mating', 'savings_rate', 'interest_rate']
        for (single_parameter_value, parameter_name) in zip(sample_point, parameter_names):
            change_parameter_in_yaml(r'C:\Users\Shaheen.Ahmed\Desktop\Utrecht_19_20\Thesis\Numerical Experiments\Thesis_modules\inheritance\model.yaml', parameter_name, float(single_parameter_value))
            print ('parameter_change_done')
        #! python adapter.py model.yaml
        subprocess.run("python adapter.py model.yaml", shell=True)
        print ('model run complete')
        data = pd.HDFStore(r'C:\Users\Shaheen.Ahmed\Desktop\Utrecht_19_20\Thesis\Numerical Experiments\Thesis_modules\inheritance\output.h5')
        df_data = data.get(r'\data')
        gini_numpy_array = df_data.loc[:,'gini'].values
        fitness_function_of_gini[sample_point_ticker] = np.mean(fitness_function_of_gini[45:50]) # Fitness function defined here
        data.close()
        sample_point_ticker += 1  
    SA.plot_1_dim_SA(used_parameter_settings[0], fitness_function_of_gini)
    return used_parameter_settings, fitness_function_of_gini


def run_SI_model(parameter_array, length_time_series):
    '''
    This function runs the Sandtable Inheritance model, and outputs a length of gini values. 
    Note:
        Only three parameters, 'assortative mating', 'savings rate' and 'interest rate' are currently able to be adjusted. 
        Delete output.h5 file before running, seems to be buggy if old outputs left. 
        I also don't think we need to generalise this function, as we will probably have to write such a function for each external model. 
        
    Inputs:
    parameter_array = numpy array (of length 3 currently), with values of three above parameters, to place in model.yaml file. 
    length_time_series = redundant in this implementation, a feature of calculate_and_plot_param_settings_vs_MSM_fn_n_dim, which seeks to take any model running function, with length of time series and parameters as inputs. 

    Output: 
    gini_numpy_array = numpy array of gini time series, currently first 80 values. 
    '''    
    print (os.getcwd())
    os.chdir(r'C:\Users\Shaheen.Ahmed\Desktop\Utrecht_19_20\Thesis\Numerical Experiments\Thesis_modules\inheritance')
    subprocess.run("python data.py") 
    #os.chdir(r'C:\Users\Shaheen.Ahmed\Desktop\Utrecht_19_20\Thesis\Numerical Experiments')
    print (os.getcwd())


    parameter_names = ['assortative_mating', 'savings_rate', 'interest_rate']
    for (single_parameter_value, parameter_name) in zip(parameter_array, parameter_names):
        change_parameter_in_yaml(r'C:\Users\Shaheen.Ahmed\Desktop\Utrecht_19_20\Thesis\Numerical Experiments\Thesis_modules\inheritance\model.yaml', parameter_name, float(single_parameter_value))
        print ('parameter_change_done')
    #! python adapter.py model.yaml
    print (os.getcwd())

    subprocess.run("python adapter.py model.yaml")
    print ('model run complete')
    data = pd.HDFStore(r'C:\Users\Shaheen.Ahmed\Desktop\Utrecht_19_20\Thesis\Numerical Experiments\Thesis_modules\inheritance\output.h5')
    df_data = data.get('/data')
    gini_numpy_array = df_data.loc[:,'gini'].values    
    gini_numpy_array = gini_numpy_array[~np.isnan(gini_numpy_array)] # Remove trailing NaNs from gini time series
    gini_numpy_array = np.delete(gini_numpy_array, np.where(gini_numpy_array == -1)) # Remove trailing -1s from gini time series
    gini_numpy_array = gini_numpy_array[0:80]
    data.close()
  
    return gini_numpy_array

