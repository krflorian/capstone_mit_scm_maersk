#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:40:18 2019

@author: Florian Krempl 

Capstone Project Predict Transit time with machine learning
"""

# first load test statistics and model!!
import os
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib
os.chdir('E:\MIT\Capstone')

# SETUP next 3 lines take some time...
exec(open('scripts/functions.py').read())
exec(open('scripts/setup_stats.py').read())

rf_booked = joblib.load('data/models/randomforest_booked_1_3.joblib')

# read new customer data
customer_clean = pd.read_excel('data/new_transport.xlsx')

# load chinese new year dates
chinese_holidays = pd.read_csv('data/statistics/chinese_holidays_complete.csv',
                               sep = ',', encoding= 'latin1')
chinese_holidays['date'] = pd.to_datetime(chinese_holidays['date'])
customer_clean['holiday'] = np.where(np.isin(customer_clean['Expected Receipt Date'],
                                              chinese_holidays['date']), 1, 0)

# get statistics
new_transports = get_features_booking(customer_clean)

# predict transit time
y_hat = rf_booked.predict(new_transports)

# calculate arrival time
for transport in range(len(y_hat)):
    customer_clean['Container Unload From Vessel-Estimated'] = customer_clean['Book Date'][transport] + datetime.timedelta(days=y_hat[transport])
    print(customer_clean[['Carrier', 'Shipper',
                          'Original Port Of Loading', 'Final Port Of Discharge',
                          'Container Unload From Vessel-Estimated']].loc[transport])

#pd.DataFrame(stat_pod['customer'].unique()).to_csv('data/shipper.csv')

    