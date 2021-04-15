from numpy.random import normal
from numpy.random import seed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import Thesis_modules.All_thesis_functions.Plotting as Plotting
import time 
import importlib
importlib.reload(plt)
importlib.reload(mpl)

def generate_ar1_series(parameter_vector, length, x_0 = 0, parameter_names = None, seed_for_KS = None):
    '''
    This function outputs an AR(1) time series.
    
    Inputs:
    length = length of time series 
    parameter_vector = numpy array of all parameters in model ([0] == alpha)
    x_0 = initial value in time series
    Outputs:
    numpy array, length = length, of AR(1) process
    '''
    alpha = parameter_vector[0] 
    ar1_data = np.empty(length)
    ar1_data[0] = x_0
    for i in range(1, length):
        epsilon_t = normal(0, 1.0) # Noise has mean zero, std. dev one
        ar1_data[i] = alpha*ar1_data[i-1] + epsilon_t   
    return ar1_data

def plot_ar1_series(ar1_series_1, width, tex_fonts):

    mpl.rcParams.update(tex_fonts)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data_print_filename = r'Figures\AR_1\thesis_figure_AR_' + timestr + r'.pdf'
    plt.figure(figsize = Plotting.set_size(width))
    plt.plot(ar1_series_1, linewidth = 0.5, color = 'k')

    plt.xlabel('Timestep')
    plt.ylabel(r'$x_t $')

    plt.title(f'Example AR(1) Time Series')
    plt.savefig(data_print_filename, format = 'pdf', bbox_inches='tight')

    #plt.show()
