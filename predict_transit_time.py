#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:40:18 2019

@author: Florian Krempl 

Capstone Project Predict Transit time with machine learning
"""

# import libraries
import os
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib
os.chdir('E:\MIT\Capstone')

# SETUP next 3 lines take some time...
# load functions
if 'stat' not in locals():
    exec(open('scripts/functions.py').read())
    # load carrier and port statistics
    exec(open('scripts/setup_stats.py').read())

model_not_ready = True
while model_not_ready:
    # load random forest model
    model = input('please input the model name you want to predict \n  booked \n received \n gate_in \n depart \n')
    filename = 'data/models/randomforest_' + model +'_new.joblib'
    
    rf = joblib.load(filename)
    
    # read new customer data (could be call to database or imput through website)
    customer_clean = pd.read_excel('data/new_transport.xlsx')
    # get date format
    date_columns = ['Expected Receipt Date', 'Book Date', 'ATD', 'ETD','ETA',
                    'Latest Receipt Date', 'Actual Receipt Date', 'Gate In Origin-Actual']
    
    for column in date_columns:
        print('start converting column', column, 'to datetime...')
        customer_clean[column] = pd.to_datetime(customer_clean[column],
                                                format = '%Y-%m-%d')
        
    # load chinese new year dates
    chinese_holidays = pd.read_csv('data/statistics/chinese_holidays_complete.csv',
                                   sep = ',', encoding= 'latin1')
    chinese_holidays['date'] = pd.to_datetime(chinese_holidays['date'])
    customer_clean['holiday'] = np.where(np.isin(customer_clean['Expected Receipt Date'],
                                                 chinese_holidays['date']), 1, 0)

    print('start to load model data...')
    if (model == 'depart'):
    # get model departed
        customer_clean = get_cap(customer_clean)
        new_transports, features = get_features_departed(customer_clean)
        error_bounds = results_depart
        model_not_ready = False
        print('model depart ready...')
        # get model received
    elif (model == 'received'):
        new_transports, features = get_features_received(customer_clean)
        error_bounds = results_received
        model_not_ready = False
        print('model received ready...')
    # get model gate_in
    elif (model == 'gate_in'):
        new_transports, features = get_features_gate(customer_clean)
        error_bounds = results_gate
        model_not_ready = False
        print('model gate_in ready...')
    # get variables booked
    elif (model == 'booked'):
        new_transports, features = get_features_booking(customer_clean)
        error_bounds = results_booking
        model_not_ready = False
        print('model booked ready...')
    # check if model is loaded
    else: 
        print('please input correct model name')
        
###############################################################################
        ## not ready !!!!!!!!!!!!!!
# get error bounds
new_transports = new_transports.merge(error_bounds,
                                      how = 'left',
                                      on=['Carrier', 'Original Port Of Loading', 'Final Port Of Discharge'])

# predict transit time
y_hat = rf.predict(new_transports[features])
new_transports['Container Unload From Vessel-Estimated'] = 'nan'
new_transports['latest'] = 'nan'
new_transports['earliest'] = 'nan'
new_transports['y_hat'] = y_hat

# calculate arrival time
for transport in range(len(y_hat)):
    if model == 'booked':
            new_transports['Container Unload From Vessel-Estimated'][transport] = new_transports['Expected Receipt Date'][transport] + datetime.timedelta(days=round(new_transports['y_hat'][transport]))
    elif model == 'depart':
            new_transports['Container Unload From Vessel-Estimated'][transport] = new_transports['ATD'][transport] + datetime.timedelta(days=round(new_transports['y_hat'][transport]))
    elif model == 'gate_in':
            new_transports['Container Unload From Vessel-Estimated'][transport] = new_transports['Gate In Origin-Actual'][transport] + datetime.timedelta(days=round(new_transports['y_hat'][transport]))
    elif model == 'received':
            new_transports['Container Unload From Vessel-Estimated'][transport] = new_transports['Actual Receipt Date'][transport] + datetime.timedelta(days=round(new_transports['y_hat'][transport]))
    
    if np.isnan(new_transports[0.9][transport]):
        print(new_transports[['Carrier', 'Shipper', 'Original Port Of Loading', 'Final Port Of Discharge',
                              'earliest', 'Container Unload From Vessel-Estimated', 'latest']].loc[transport])
        continue
    new_transports['latest'][transport] = new_transports['Container Unload From Vessel-Estimated'][transport] + datetime.timedelta(days=round(new_transports[0.9][transport]))
    new_transports['earliest'][transport] = new_transports['Container Unload From Vessel-Estimated'][transport] + datetime.timedelta(days=round(new_transports[0.1][transport]))
    # print ETA with error bounds
    print(new_transports[['Carrier', 'Shipper', 'Original Port Of Loading', 'Final Port Of Discharge',
                          'earliest', 'Container Unload From Vessel-Estimated', 'latest']].loc[transport])

new_transports.columns
