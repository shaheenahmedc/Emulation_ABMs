import matplotlib.pyplot as plt

def plot_1_dim_SA(used_parameter_settings, fitness_for_each_parameter_setting):
    '''
    This function plots a 1d fitness surface, for one varying parameter.
    
    Inputs:
    used_parameter_settings = (1 x (number of sample points - number of NANs in fitness function)) shape numpy array, of parameter to vary.  
    fitness_for_each_parameter_setting = (number of sample points - number of NANs in fitness function) shape numpy array, of non-NAN fitness values. 
    
    Output:
    1d fitness function plot. 
    '''
    
    plt.figure()
    plt.scatter(used_parameter_settings, fitness_for_each_parameter_setting, s = 2)
    plt.xlabel("Parameter value")
    plt.ylabel("fitness function value")
    plt.show()