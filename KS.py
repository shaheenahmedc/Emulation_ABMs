# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:39:47 2021

@author: shahe
"""

import csv
import numpy as np
import subprocess
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
def generate_KS_dGDP_time_series(param_vector, length_time_series, parameter_names = [], seed_for_KS = None):
    '''
    inputs:
        param_vector - 58 long, names provided here, to prevent changing current calibration code. 
        length_time_series
    output:
        time-series of differences in GDP, from KS 0.1 model. 
    '''
    
    # Pseudocode
    # Take param_vector, and write csv with it, newconf.csv
    # Needs to account for variable number of params
    #print (parameter_names)
    parameter_names_copy = parameter_names.copy()
    parameter_names_copy.insert(0, 'Par')
    param_vector = np.append(param_vector, seed_for_KS)
    
    print (f'param_vector in KS = {param_vector}')
    #['Par', 'Crec', 'gG', 'mLim', 'tr', 'Lambda', 'Lambda0', 'r', 'L1rdMax', 'L1shortMax', 'NW10', 'Phi3', 'Phi4', 'alpha1', 'beta1', 'alpha2', 'beta2', 'd1', 'gamma', 'm1', 'mu1', 'nu', 'x1inf', 'x1sup', 'x5', 'xi', 'zeta1', 'zeta2', 'NW20', 'Phi1', 'Phi2', 'chi', 'd2', 'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'f2min', 'iota', 'kappaMax', 'kappaMin', 'm2', 'mu20', 'omega1', 'omega2', 'u', 'upsilon', 'delta', 'phi', 'psi1', 'psi2', 'psi3', 'w0min']
    #parameter_names = ['Par', 'chi']
    #chi_value = param_vector[30]
    #param_vector = np.array([chi_value])
    param_values_list = list(param_vector)
    param_values_list.insert(0, 'Cfg1')
    rows = zip(parameter_names_copy, param_values_list)
    row_list = []
    row_list.append(parameter_names_copy)
    row_list.append(param_values_list)
    
    #print (f'rows = {rows}')
    
    with open('nextconf.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            #print (f'row = {row}')
            writer.writerow(row)
    print ('nextconf.csv writing complete')
    # Change benchmark with newconf.csv:

    #os.system("PATH C:\\Users\shahe\\LSD\\gnu\\bin;%PATH% & lsdNW -t -z -f Benchmark_gdp_outputs.lsd")
    #os.system("PATH C:\\Users\shahe\\LSD\\gnu\\bin;%PATH% & lsdNW -t -z -f nextconf.lsd")
    
    #os.system("PATH C:\\Users\shahe\\LSD\\gnu\\bin;%PATH% & C:\\Users\\shahe\\LSD\\lwi\\lsd_confgen -f Benchmark_gdp_outputs_16_march_2021.lsd -c nextconf.csv -o nextconf & lsdNW -t -z -f nextconf.lsd")
    os.system("PATH C:\\Users\shahe\\LSD\\gnu\\bin;%PATH% & C:\\Users\\shahe\\LSD\\lwi\\lsd_confgen -f Benchmark_gdp_outputs_16_march_2021.lsd -c nextconf.csv -o nextconf & lsdNW -c 1 -t -z -f nextconf.lsd")
    print ('os command executed')

    # Read in resultant csv file
    output_file_path = 'nextconf_' + str(seed_for_KS) + '.csv'
    print (f'output_file_path= {output_file_path}')
    output_data = pd.read_csv(output_file_path)
    print ('csv read complete')

    #Isolate dGDP time-series, return as numpy array 
    dGDP_data = output_data['dGDP'].values
    #print (f'dGDP_data = {dGDP_data}')
    #GDP_data = output_data['GDP'].values
    #print (f'GDP_data = {GDP_data}')

    #plt.figure()
    #plt.plot(GDP_data)
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    #plt.savefig(r'C:\Users\shahe\OneDrive\Documents\Utrecht_19_20\Thesis\Numerical_Experiments\Figures\KS\GDP_KS_' + timestr + r'.pdf', format = 'pdf', bbox_inches='tight')
    #plt.close()
    print ('entire KS function complete')

    #print (f'dGDP_data = {dGDP_data}')
    return dGDP_data


def return_all_KS_init_arrays(indices_params_to_vary):
    n_country_params = 1
    # [Name, lower_bound, init, upper_bound]
    tr_info = [r'$tr$', 0.0, 0.1, 0.3]
    
    country_parameter_names = [tr_info[0]]
    lower_bounds_country_params = np.array([tr_info[1]])
    init_values_country_params = np.array([tr_info[2]])
    upper_bounds_country_params = np.array([tr_info[3]])
    
    
    n_financial_market_params = 2
    Lambda_info = [r'$Lambda$', 0.01, 2.0, 4.0]
    r_info = [r'$r$', 0.005, 0.01, 0.05]
    
    financial_market_parameter_names = [Lambda_info[0], r_info[0]]
    lower_bounds_financial_market_params = np.array([Lambda_info[1], r_info[1]])
    init_values_financial_market_params = np.array([Lambda_info[2], r_info[2]])
    upper_bounds_financial_market_params = np.array([Lambda_info[3], r_info[3]])
    
    
    n_capital_market_params = 16
    NW10_info = [r'$NW10$', 200, 1000, 5000]
    Phi3_info = [r'$Phi3$', 0.0, 0.1, 0.50]
    Phi4_info = [r'$Phi4$', 0.50, 0.9, 1.0]
    alpha1_info = [r'$alpha1$', 1.0, 3.0, 5.0]
    beta1_info = [r'$beta1$', 1.0, 3.0, 5.0]
    alpha2_info = [r'$alpha2$', 1.0, 2.0, 5.0]
    beta2_info = [r'$beta2$', 1.0, 4.0, 5.0]
    gamma_info = [r'$gamma$', 0.2, 0.5, 0.8]
    mu1_info = [r'$mu1$', 0.01, 0.08, 0.2]
    nu_info = [r'$nu$', 0.01, 0.04, 0.2]
    x1inf_info = [r'$x1inf$', -0.3, -0.15, -0.1]
    x1sup_info = [r'$x1sup$', 0.1, 0.15, 0.30]
    x5_info = [r'$x5$', 0.0, 0.3, 1.0]
    xi_info = [r'$xi$', 0.2, 0.5, 0.8]
    zeta1_info = [r'$zeta1$', 0.1, 0.3, 0.6]
    zeta2_info = [r'$zeta2$', 0.1, 0.3, 0.6]
        
    capital_market_parameter_names = [
                       NW10_info[0],
                       Phi3_info[0],
                       Phi4_info[0],
                       alpha1_info[0],
                       beta1_info[0],
                       alpha2_info[0],
                       beta2_info[0],
                       gamma_info[0],
                       mu1_info[0],
                       nu_info[0],
                       x1inf_info[0],
                       x1sup_info[0],
                       x5_info[0],
                       xi_info[0],
                       zeta1_info[0],
                       zeta2_info[0]
                      ]
    
    lower_bounds_capital_market_params = np.array([
                       NW10_info[1],
                       Phi3_info[1],
                       Phi4_info[1],
                       alpha1_info[1],
                       beta1_info[1],
                       alpha2_info[1],
                       beta2_info[1],
                       gamma_info[1],
                       mu1_info[1],
                       nu_info[1],
                       x1inf_info[1],
                       x1sup_info[1],
                       x5_info[1],
                       xi_info[1],
                       zeta1_info[1],
                       zeta2_info[1]
                      ])
        
    init_values_capital_market_params = np.array([
                       NW10_info[2],
                       Phi3_info[2],
                       Phi4_info[2],
                       alpha1_info[2],
                       beta1_info[2],
                       alpha2_info[2],
                       beta2_info[2],
                       gamma_info[2],
                       mu1_info[2],
                       nu_info[2],
                       x1inf_info[2],
                       x1sup_info[2],
                       x5_info[2],
                       xi_info[2],
                       zeta1_info[2],
                       zeta2_info[2]
                      ])
    
    upper_bounds_capital_market_params = np.array([
                       NW10_info[3],
                       Phi3_info[3],
                       Phi4_info[3],
                       alpha1_info[3],
                       beta1_info[3],
                       alpha2_info[3],
                       beta2_info[3],
                       gamma_info[3],
                       mu1_info[3],
                       nu_info[3],
                       x1inf_info[3],
                       x1sup_info[3],
                       x5_info[3],
                       xi_info[3],
                       zeta1_info[3],
                       zeta2_info[3]
                      ])
    
    
    n_consumer_market_params = 12
        
    NW20_info = [r'$NW20$', 200, 1000, 5000]
    Phi1_info = [r'$Phi1$', 0.0, 0.1, 0.50]
    Phi2_info = [r'$Phi2$', 0.50, 0.9, 1.0]
    chi_info = [r'$chi$', 0.2, 1.0, 5.0]
    f2min_info = [r'$f2min$', 1e-06, 1e-05, 0.001]
    iota_info = [r'$iota$', 0.0, 0.1, 0.3]
    m2_info = [r'$m2$', 10.0, 40.0, 100.0]
    mu20_info = [r'$mu20$', 0.1, 0.3, 0.5]
    omega1_info = [r'$omega1$', 0.2, 1.0, 5.0]
    omega2_info = [r'$omega2$', 0.2, 1.0, 5.0]
    u_info = [r'$u$', 0.5, 0.75, 1.0]
    upsilon_info = [r'$upsilon$', 0.01, 0.04, 0.1]
    
    consumer_market_parameter_names = [
                       NW20_info[0],
                       Phi1_info[0],
                       Phi2_info[0],
                       chi_info[0],
                       f2min_info[0],
                       iota_info[0],
                       m2_info[0],
                       mu20_info[0],
                       omega1_info[0],
                       omega2_info[0],
                       u_info[0],
                       upsilon_info[0]                       
                      ]
    
    lower_bounds_consumer_market_params = np.array([
                       NW20_info[1],
                       Phi1_info[1],
                       Phi2_info[1],
                       chi_info[1],
                       f2min_info[1],
                       iota_info[1],
                       m2_info[1],
                       mu20_info[1],
                       omega1_info[1],
                       omega2_info[1],
                       u_info[1],
                       upsilon_info[1]                       
                      ])
        
    init_values_consumer_market_params = np.array([
                       NW20_info[2],
                       Phi1_info[2],
                       Phi2_info[2],
                       chi_info[2],
                       f2min_info[2],
                       iota_info[2],
                       m2_info[2],
                       mu20_info[2],
                       omega1_info[2],
                       omega2_info[2],
                       u_info[2],
                       upsilon_info[2]                       
                      ])
    
    upper_bounds_consumer_market_params = np.array([
                       NW20_info[3],
                       Phi1_info[3],
                       Phi2_info[3],
                       chi_info[3],
                       f2min_info[3],
                       iota_info[3],
                       m2_info[3],
                       mu20_info[3],
                       omega1_info[3],
                       omega2_info[3],
                       u_info[3],
                       upsilon_info[3]                       
                      ])
    
    n_labor_supply_params = 2
                 
    phi_info = [r'$phi$', 0.0, 0.4, 1.0]
    psi2_info = [r'$psi2$', 0.95, 1.0, 1.05]
                
    labor_supply_parameter_names = [
                       phi_info[0],
                       psi2_info[0]
                      ]
                 
    lower_bounds_labor_supply_params = np.array([
                       phi_info[1],
                       psi2_info[1]
                      ])
        
    init_values_labor_supply_params = np.array([
                       phi_info[2],
                       psi2_info[2]
                      ])
    
    upper_bounds_labor_supply_params = np.array([
                       phi_info[3],
                       psi2_info[3]
                      ])
    
    all_parameter_names = np.concatenate((country_parameter_names, 
                                           financial_market_parameter_names,
                                           capital_market_parameter_names, 
                                           consumer_market_parameter_names,
                                           labor_supply_parameter_names))
    
    all_param_names_stripped_of_dollar = [i.replace('$', '') for i in all_parameter_names]
    
    parameter_lower_bounds_vector = np.concatenate((lower_bounds_country_params, 
                                                   lower_bounds_financial_market_params,
                                                   lower_bounds_capital_market_params, 
                                                   lower_bounds_consumer_market_params,
                                                   lower_bounds_labor_supply_params))
    
    
    parameter_upper_bounds_vector = np.concatenate((upper_bounds_country_params, 
                                                   upper_bounds_financial_market_params,
                                                   upper_bounds_capital_market_params, 
                                                   upper_bounds_consumer_market_params,
                                                   upper_bounds_labor_supply_params))
    
    init_values_vector = np.concatenate((init_values_country_params, 
                                                   init_values_financial_market_params,
                                                   init_values_capital_market_params, 
                                                   init_values_consumer_market_params,
                                                   init_values_labor_supply_params))
    
    vector_non_randomised_parameter_values = init_values_vector.copy()

    vector_non_randomised_parameter_values[indices_params_to_vary] = np.inf
    
    return all_param_names_stripped_of_dollar, parameter_lower_bounds_vector, parameter_upper_bounds_vector, init_values_vector, vector_non_randomised_parameter_values