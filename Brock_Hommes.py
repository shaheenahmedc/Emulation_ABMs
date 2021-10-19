import numpy as np 
from numpy.random import normal
import matplotlib.pyplot as plt

def generate_Brock_Hommes_time_series(params_vector, len_series, parameter_names = None, seed_for_KS = None):
    '''
    This function outputs a time series from the 1998 Brock and Hommes model, as detailed in https://www.sciencedirect.com/science/article/pii/S0165188920300294. 
    
    Parameters
    ----------
    params_vector: a numpy array containing all the parameters to run the model for. In the following order:
        [g_vector, b_vector, beta, r, sigma]
        This is done to allow a general MSM function to be written, which can take any model function with the same two parameters. 
    g_vector: vector of trend-following parameters, of length(number of strategies)
    b_vector: vector of bias parameters, of length(number of strategies)
    beta: switching intensity parameter
    r: market interest rate
    sigma: noise standard deviation
    len_series: desired length of output time series
    parameter_names: bad design. In Calibration.py, model_func is our general data generating process, but KS data generator needs four parameters, 
        while AR, BH and FW only need 2. So I had to include these unused parameters here. Fix later. 
    seed_for_KS: same as parameter_names. Bad design, figure out how to remove.
    
    Outputs
    ----------
    output: time series of length(len_series)
    '''
    params_vector = params_vector.flatten()
    g_vector = params_vector[0:4]
    b_vector = params_vector[4:8]
    beta = params_vector[8]
    r = params_vector[9]
    sigma = params_vector[10]
    R = 1.0 + r # As defined in Platt_20
    
    if (len(g_vector) != len(b_vector)):
        print ("ERROR: g_vector and b_vector different lengths") # Trend following parameter set and bias parameter set should have same number of values
        
    U = np.zeros([len(g_vector), len_series]) # Initialise U array, to hold U_h_t values
    n = np.zeros([len(g_vector), len_series]) # Initialise n array, to hold n_h_t values
    output = np.zeros(len_series)
    for i in range(2, len_series - 1): # We start at the third datapoint in time series, as U_h_t depends on previous two entries
        U_max = U[:,i].max()
        U[:, i] = ((output[i] - R*output[i-1]) * (g_vector * output[i-2] + b_vector - R * output[i-1]))
        n[:, i+1] = np.exp((beta * U[:,i]) - U_max) / np.sum(np.exp((beta * U[:,i]) - U_max))

        # Redundant calculation below. Delete when confident no longer needed. Some info on underflow/overflow treatment in hashed code. 
        '''
        for j in range (0, len(g_vector)): # Loop over strategies at each timestep
            U[j, i] = (output[i] - R*output[i-1]) * (g_vector[j]*output[i-2] + b_vector[j] - R*output[i-1])
            #print (f' U[j, i] = { U[j, i]}') 
            
            # We need to be careful of underflow/overflow. Here its overflow, as we exponentiate:
            # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            n[j, i+1] = np.exp((beta * U[j, i]) - U_max) / np.sum(np.exp((beta * U[:,i]) - U_max))
        '''
        
        #epsilon_t_vector = normal(0, sigma**2, size = len(g_vector))
        epsilon_t = normal(0, sigma**2)
        
        #output[i+1] = (1.0/R) * (np.sum(n[:, i+1]*(g_vector*output[i] + b_vector)+ epsilon_t_vector)) # Double check if this noise implementation correct, inside or outside sum?
        output[i+1] = (1.0/R) * (np.sum(n[:, i+1]*(g_vector*output[i] + b_vector)) + epsilon_t) # Double check if this noise implementation correct, inside or outside sum?
    
    return output

