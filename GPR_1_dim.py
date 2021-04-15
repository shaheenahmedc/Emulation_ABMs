# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:37:29 2020

@author: Shaheen.Ahmed
"""

import numpy as np 
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

def kernel(x1, x2, length_scale = 1.0, variance = 1.0):
    '''
    Isotropic squared exponential kernel.
    Computes a covariance matrix from points in x1 and x2.
    
    Args:
    x1: Array of m points (m x d).
    x2: Array of n points (n x d). 
    
    Returns:
    Covariance matrix (m x n):
    '''
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T) #returns matrix of euclidean distances between points
    return variance * np.exp((-0.5 / length_scale**2) * sqdist)

def plot_gp(mu, cov, x, x_train, y_train, GPR_fit_title, GPR_fit_filename, samples=[]):
    '''
    This function plots a GPR surface, over a given domain x, interpolating between points x_train. 
    
    Inputs:
    mu = mean vector mu(x) for input GPR.
    cov = covariance matrix for input GPR. 
    x = range of GPR predictions. 
    x_train = points to interpolate between.
    y_train = values of fitness function for x_train. 
    samples = redundant, remove
    length_scale = (usually optimised) length scale
    kernel_variance = same
    noise = same
    '''
    #print ("function accessed!")
    x = x.ravel()
    mu = mu.ravel()
    
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.figure(figsize = (10,7))
    plt.fill_between(x, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(x, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(x, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if x_train is not None:
        plt.plot(x_train, y_train, 'rx')
    plt.xlabel('Parameter value')
    plt.ylabel('Fitness value')
    plt.title(GPR_fit_title)
    plt.savefig(GPR_fit_filename)
    plt.show()
    
def plot_covariance_heatmap(cov):
    '''
    This function plots the covariance matrix as a heatmap. 
    Inputs:
    cov = GPR covariance matrix.
    '''
    plt.figure()
    plt.imshow(cov, cmap='hot')
    plt.title("Entries in Covariance Matrix, colourised")
    plt.colorbar()
    plt.show()
    
def GP_posterior_mean_and_cov(x_test, x_train, y_train, length_scale = 1.0, kernel_variance = 1.0 , noise = 1.0):
    '''
    Computes posterior mean mu(x) and covariance function cov(x) of the GP, given the observation of m training data x_train 
    and y_train, and n new inputs x_test.
    
    Args:
    x_test: new(test) input variables/locations (n x d).
    x_train: training input variables/locations (m x d).
    y_train: training output variables (m x 1).
    length_scale: kernel length scale parameter.
    kernel_variance: kernel vertical variation parameter.
    noise: noise parameter.
    
    Returns:
    Posterior mean vector (n x d).
    Covariance matrix (n x n).
    '''
    
    # Calculate kernel(training data, training data), kernel(training data, test data), 
    # kernel (test data, training data), and kernel(test_data, test_data)
    K_train_train = kernel(x_train, x_train, length_scale, kernel_variance) + noise**2 * np.eye(len(x_train)) # K = k(x, x) + sigma^2(Identity Matrix)
    K_train_test = kernel(x_train, x_test, length_scale, kernel_variance)
    K_test_test = kernel(x_test, x_test, length_scale, kernel_variance) + noise * np.eye(len(x_test)) #Changing the constant here to vary with noise changes results dramatically for noisy GPR. Which to use? 
    K_train_train_inv = inv(K_train_train)
    
    # Conditional mean
    mu_test = K_train_test.T.dot(K_train_train_inv).dot(y_train)
    # Conditional covariance matrix
    cov_test = K_test_test - K_train_test.T.dot(K_train_train_inv).dot(K_train_test)
    
    return mu_test, cov_test

def nll_fn(x_train, y_train, naive = False):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given 
    noise level.
    
    Args:
    x_train: training locations (m x d).
    y_train: training targets (m x 1).
    naive: if True use a naive implementation of Eq. (7), if False use a numerically more stable implementation. 
        
    Returns:
        Minimization objective.
    '''
    def nll_naive(theta):
        # Naive implementation of Eq. (7) in http://krasserm.github.io/2018/03/19/gaussian-processes/. 
        #Works well for the examples in this article but is numerically less stable compared to 
        # the implementation in nll_stable below.
        K = kernel(x_train, x_train, length_scale = theta[0], variance = theta[1]) + theta[2]**2 * np.eye(len(x_train))
        return 0.5 * np.log(det(K)) + 0.5 * y_train.T.dot(inv(K).dot(y_train)) + 0.5 * len(x_train) * np.log(2*np.pi)

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        K = kernel(x_train, x_train, length_scale = theta[0], variance = theta[1]) + theta[2]**2 * np.eye(len(x_train))
        L = cholesky(K)
        nll_equation = np.sum(np.log(np.diagonal(L))) + 0.5 * y_train.T.dot(lstsq(L.T, lstsq(L, y_train)[0])[0]) + 0.5 * len(x_train) * np.log(2*np.pi)
        # For some reason, the numpy array that results from nll_equation has dimension two
        # And looks like [[result]]
        # We want [result], because minimize needs the objective function's output to be one-dimension
        # So we use np.squeeze on it
        return np.squeeze(nll_equation)
        
        #return 10*theta[0] -4*theta[1]
    if naive:
        return nll_naive
    else:
        return nll_stable
    
def GPR_wrapper(x, 
                x_train, 
                y_train,           
                init_conds_length_scale_kernel_var_noise, 
                bounds_length_scale_kernel_var_noise,
                naive = False,
                method='L-BFGS-B'
               ):
    '''
    Returns a plot of GPR on the I/O pairs given.
    Optimises length scale and kernel variance, but not noise. To do. 
    
    Args:
    x: range to run GPR over
    x_train: training locations (m x d).
    y_train: training targets (m x 1).
    init_conds_length_scale_kernel_var_noise: Initial conditions for GPR hyperparameter optimisation.
    bounds_length_scale_kernel_var_noise: Bounds within which hyperparams are allowed to vary in optimisation. 
    naive: if True use a naive implementation of Eq. log likelihood, if False use a numerically more stable implementation. 
    method: optimisation method.
    
    Returns:
        Plot of GPR on I/O pairs. 
    '''
    res = minimize(nll_fn(x_train, y_train), 
                   init_conds_length_scale_kernel_var_noise, 
                   bounds=(bounds_length_scale_kernel_var_noise, 
                           bounds_length_scale_kernel_var_noise, 
                           bounds_length_scale_kernel_var_noise), 
                   method = method)
    l_opt, sigma_f_opt, noise_opt = res.x
    mu_test, cov_test = GP_posterior_mean_and_cov(x, x_train, y_train, length_scale = l_opt, kernel_variance = sigma_f_opt, noise = noise_opt)
    plot_gp(mu_test, cov_test, x, x_train = x_train, y_train = y_train , length_scale = l_opt, kernel_variance = sigma_f_opt, noise = noise_opt)
    return mu_test

def GPR_wrapper_sklearn_sq_exp_kernel_plus_noise(
                       used_parameter_settings,
                       MSM_value_for_each_parameter_setting
                       ):
    '''
    This function implements GPR via sklearn, with a simple squared exponential kernel plus noise. 
    length_scale, kernel variance and noise are once again optimised over.
    The resulting GPR surface is then plotted.
    
    Inputs:
    used_parameter_settings = numpy array of used parameter settings.
    MSM_value_for_each_parameter_setting = MSM value for each parameter setting, averaged over repeated runs at each parameter setting.
    
    Outputs:
    mu_test = GPR predictions at used_parameter_settings 
    '''
    # ConstantKernel represents kernel_variance in our manual GPR implementation
    # WhiteKernel represents our noise parameter, can also be set as alpha in GaussianProcessRegressor I think
    # But I don't think it can be optimised there
    # Some debate over this though? If it's the same?
    # https://stackoverflow.com/questions/54987985/using-scikit-learns-whitekernel-for-gaussian-process-regression
    rbf = ConstantKernel(1000) * RBF(length_scale=1.0) + WhiteKernel(noise_level = 10) 
    gpr = GaussianProcessRegressor(kernel=rbf) # alpha = noise parameter as above, not the kernel variance

    # Reuse training data from previous 1D example
    gpr.fit(used_parameter_settings.reshape(-1,1), MSM_value_for_each_parameter_setting.reshape(-1,1))

    # Compute posterior predictive mean and covariance
    mu_test, cov_test = gpr.predict(used_parameter_settings.reshape(-1,1), return_cov=True)

    # Obtain optimized kernel parameters
    # param naming is a bit weird to get used to 
    # A product of kernels, so for us ConstantKernel is k1, RBF is k2, White

    k1_params = gpr.kernel_.k1.get_params()
    #print (k1_params)
    #print ('--------------------')
    k2_params = gpr.kernel_.k2.get_params()
    #print (k2_params)
    #print ('--------------------')

    # If we want the kernel_variance, we get it like this:
    kernel_variance = np.sqrt(gpr.kernel_.k1.get_params()['k1__constant_value'])

    # If we want the length_scale, we get it like this:
    # Very odd, we have to reference the k1 kernel, but the param is called k2_length_scale? 
    length_scale = gpr.kernel_.k1.get_params()['k2__length_scale']

    # If we want the noise_parameter, we get it like this:
    noise_parameter = np.sqrt(gpr.kernel_.k2.get_params()['noise_level'])
    #print (length_scale, kernel_variance, noise_parameter)
    # Compare with previous results
    #assert(np.isclose(l_opt, length_scale))
    #assert(np.isclose(sigma_f_opt, kernel_variance))

    # Plot the results
    plot_gp(mu_test, 
            cov_test, 
            used_parameter_settings, 
            x_train = used_parameter_settings, 
            y_train = MSM_value_for_each_parameter_setting,
            length_scale = length_scale, 
            kernel_variance = kernel_variance,
            noise = noise_parameter)
            
    return mu_test

def GPR_wrapper_gpy_sq_exp_kernel_plus_noise(used_parameter_settings,
                       MSM_value_for_each_parameter_setting,
                       init_conds_length_scale_kernel_var_noise = [1.0,1.0,1.0]):
    
    '''
    This function implements GPR via GPy, with a simple squared exponential kernel plus noise. 
    length_scale, kernel variance and noise are once again optimised over.
    The resulting GPR surface is then plotted.
    
    Inputs:
    used_parameter_settings = numpy array of used parameter settings.
    MSM_value_for_each_parameter_setting = MSM value for each parameter setting, averaged over repeated runs at each parameter setting.
    
    Outputs:
    mu_vector_from_x_values = GPR predictions at used_parameter_settings.
    '''
    noise = init_conds_length_scale_kernel_var_noise[2]
    rbf = GPy.kern.RBF(input_dim = 1, variance = init_conds_length_scale_kernel_var_noise[1], lengthscale = init_conds_length_scale_kernel_var_noise[0])
    gpr = GPy.models.GPRegression(used_parameter_settings.reshape(-1,1), MSM_value_for_each_parameter_setting.reshape(-1,1), rbf)

    # Fix the noise variance to known value 
    gpr.Gaussian_noise.variance = noise**2
    #gpr.Gaussian_noise.variance.fix()

    # Run optimization
    gpr.optimize();
    # Obtain optimized kernel parameters
    length_scale = gpr.rbf.lengthscale.values[0]
    kernel_variance = np.sqrt(gpr.rbf.variance.values[0])

    # Plot the results with the built-in plot function
    gpr.plot(plot_limits = np.array([0,1]));
    mu_vector_from_x_values, __ = gpr.predict(used_parameter_settings.reshape(-1,1))
    return mu_vector_from_x_values

