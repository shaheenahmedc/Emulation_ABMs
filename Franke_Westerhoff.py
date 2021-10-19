import numpy as np
import matplotlib.pyplot as plt

def generate_Franke_Westerhoff_returns_time_series(params_vector, len_series, parameter_names = None, seed_for_KS = None):
    '''
    This function generates a time series of returns from the Franke_Westerhoff model. 
    Specifically, the DCA method is used, as opposed to the TPA method. See Franke_12 for more details. 
    We also only implement the HPM variant. 
    
    Parameters
    ----------
    params_vector is divided as follows:
        params_vector = [mu, beta, phi, chi, alpha_n, alpha_O, alpha_p, sigma_f, sigma_c]
        Each parameter is a scalar float. 
    
        'Set' parameters:
        mu = 0.01
        beta = 1
        phi = 0.12
        chi = 1.5
        sigma_f = 0.758
        
        Bounds for free parameters:
        alpha_O in [-1,1]
        alpha_p in [0,20]
        alpha_n n [0, 2]
        sigma_c in [0,5]
    
    len_series: length of time series. 
    parameter_names: bad design. In Calibration.py, model_func is our general data generating process, but KS data generator needs four parameters, 
        while AR, BH and FW only need 2. So I had to include these unused parameters here. Fix later. 
    seed_for_KS : same as parameter_names. Bad design, figure out how to remove.
    
    Outputs
    ----------
    rr_flattened: output time-series from Franke_Westerhoff model
    '''
    
    T = len_series
    P = np.zeros([T+1,1])
    pstar = 0
    Df = np.zeros([T,1])
    Dc = np.zeros([T,1])
    Nf = np.zeros([T,1])
    Nc = np.zeros([T,1])
    Gf = np.zeros([T,1])
    Gc = np.zeros([T,1])
    Wf = np.zeros([T,1])
    Wc = np.zeros([T,1])
    A = np.zeros([T,1])
    
    mu  = params_vector[0]
    beta  = params_vector[1]
    phi = params_vector[2] 
    chi  = params_vector[3] 
    alpha_n = params_vector[4]
    alpha_O  = params_vector[5] 
    alpha_p  = params_vector[6] 
    sigma_f  = params_vector[7]
    sigma_c  = params_vector[8]
        
    # Initial values(not given)
    Nf[0:2] = 0.5
    Nc[0:2] = 0.5
    
    for t in range(2,T):
        
        #portfolio performance
        #Gf[t] = ( np.exp(P[t]) - np.exp(P[t-1]) ) * Df[t-2]
        #Gc[t] = ( np.exp(P[t]) - np.exp(P[t-1]) ) * Dc[t-2]

        #summarize performance over time
        #Wf[t] = eta * Wf[t-1] + (1 - eta) * Gf[t]
        #Wc[t] = eta * Wc[t-1] + (1 - eta) * Gc[t]

        # type fractions
        Nf[t] = 1.0 / ( 1.0 + np.exp(-beta * A[t-1]))
        Nc[t] = 1 - Nf[t]

        # The A[t] dynamic is set up to handle several models
        A[t] = alpha_n * ( Nf[t] - Nc[t]) + alpha_O + alpha_p * ( pstar - P[t])**2

        # demands
        Df[t] = phi * ( pstar - P[t] ) + sigma_f * np.random.randn(1)
        Dc[t] = chi * ( P[t] - P[t-1] ) + sigma_c * np.random.randn(1)

        # pricing
        P[t+1] = P[t] + mu * ( Nf[t] * Df[t] + Nc[t] * Dc[t] )

    # returns
    rr = P[1:T+1] - P[0:T]
    rr_flattened = rr.flatten()
    
    return rr_flattened
