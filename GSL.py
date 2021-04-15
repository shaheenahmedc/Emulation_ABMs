import Thesis_modules.pygsl_div.pygsl_div as pygsl_div

def Sandtable_GSL_wrapper(pseudo_true_or_empirical_time_series, model_generated_time_series):
    '''
    This function wraps Sandtables GSL-Div implementation, to make it work for us. 
    We need to reshape our numpy arrays to (1 x len(time_series)), to make them work with the code. 
    
    Inputs:
    pseudo_true_or_empirical_time_series = (1 x len(time_series)) shape numpy array of pseudo-true or empirical data.
    model_generated_time_series = (1 x (len(time_series))) shape numpy array of pseudo-true or empirical data. 
    
    Output:
    fitness = GSL-Div value calculated between two time series. 
    '''
    pseudo_true_or_empirical_time_series = pseudo_true_or_empirical_time_series.reshape(1, len(pseudo_true_or_empirical_time_series))
    model_generated_time_series = model_generated_time_series.reshape(1, len(model_generated_time_series))
    fitness = pygsl_div.gsl_div(pseudo_true_or_empirical_time_series, model_generated_time_series, weights='add-progressive',
            b=50, L=3, min_per=1, max_per=99, state_space=[-10, 10])
    return fitness
