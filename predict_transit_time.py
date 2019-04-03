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
exec(open('scripts/functions.py').read())
# load carrier and port statistics
exec(open('scripts/setup_stats.py').read())
# load random forest model
rf_booked = joblib.load('data/models/randomforest_booked_1_4.joblib')

# read new customer data (could be call to database or imput through website)
customer_clean = pd.read_excel('data/new_transport.xlsx')
date_columns = ['Expected Receipt Date', 'Book Date']
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

# merge statistics with new transport data
new_transports = get_features_booking(customer_clean)

# get error bounds
customer_clean = customer_clean.merge(results_booking,
                                      how = 'left',
                                      on=['Carrier', 'Original Port Of Loading', 'Final Port Of Discharge'])

# predict transit time
y_hat = rf_booked.predict(new_transports)
customer_clean['Container Unload From Vessel-Estimated'] = 'nan'
customer_clean['latest'] = 'nan'
customer_clean['earliest'] = 'nan'

# calculate arrival time
for transport in range(len(y_hat)):
    customer_clean['Container Unload From Vessel-Estimated'][transport] = customer_clean['Expected Receipt Date'][transport] + datetime.timedelta(days=y_hat[transport])
    customer_clean['latest'][transport] = customer_clean['Container Unload From Vessel-Estimated'][transport] + datetime.timedelta(days=customer_clean[0.8][transport])
    customer_clean['earliest'][transport] = customer_clean['Container Unload From Vessel-Estimated'][transport] + datetime.timedelta(days=customer_clean[0.2][transport])
    # print ETA with error bounds
    print(customer_clean[['Carrier', 'Shipper', 'Original Port Of Loading', 'Final Port Of Discharge',
                          'earliest', 'Container Unload From Vessel-Estimated', 'latest']].loc[transport])

#pd.DataFrame(stat_pod['customer'].unique()).to_csv('data/shipper.csv')

