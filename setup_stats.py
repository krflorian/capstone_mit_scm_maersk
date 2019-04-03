# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:01:00 2019

@author: Florian Krempl 

Capstone Project Predict Transit time with machine learning
"""

import os
import numpy as np
import pandas as pd
import datetime

os.chdir('E:\MIT\Capstone')
os.getcwd()

customer_clean = pd.read_csv("data/customer_clean.csv", index_col = 'Unnamed: 0')

date_columns = ['Container Unload From Vessel-Actual',
                'ATA', 'ATD', 'ETA', 'ETD', 'Latest Receipt Date',
                'Book Date', 'Expected Receipt Date', 'Actual Receipt Date']

for column in date_columns:
    print('start converting column', column, 'to datetime...')
    customer_clean[column] = pd.to_datetime(customer_clean[column],
                                            format = '%Y-%m-%d')
print('begin creating new columns...')

########################################################################################

stat = customer_clean[['Original Port Of Loading', 'Final Port Of Discharge',
                       'Carrier', 'customer', 'Shipper', 'cap', 'y_depart', 
                       'Container Unload From Vessel-Actual', 'ETD', 'ETA', 'ATD', 'ATA',]]   #for departed
                
stat['route'] = (stat['ATA']-stat['ATD']).dt.days # transit time on water
stat['pod'] = (stat['Container Unload From Vessel-Actual']-stat['ATA']).dt.days # time in port of destination

#test['late_departure'] = np.where((test['ATD']-test['ETD']).dt.days > 0, 1, 0) # if vessel left after scheduled time
#test['ETP'] = (test['Expected Receipt Date'] - test['Book Date']).dt.days   # expected time between booking and receival

# sanity test
stat = stat[stat['ATD'].isnull() == False]
stat = stat[stat['cap'] > 0]
stat = stat[stat['route'] > 0]
stat = stat[(stat['pod'] <= 3) & (stat['pod'] >= 0)]
stat = stat[(stat['y_depart'] > 0) & (stat['y_depart'] < 40)]
stat = stat.reset_index(drop=True)
print('droped wrong or nan values..')

# create date columns
#stat['month'] = stat['ATD'].dt.month
stat['quarter'] = stat['ETD'].dt.quarter
#stat['arrival_day'] = stat.apply(lambda x: x['ETA'].strftime('%A'), axis=1)
stat['departure_day'] = stat.apply(lambda x: x['ATD'].strftime('%A'), axis=1)
print('got new columns...')

############################################################################################
# split into train stat for creating statistics:
date = max(stat['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
stat_train = stat[stat['Container Unload From Vessel-Actual'] < date]
# double the observations in the closest year
date2 = max(stat_train['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
stat_train = stat_train.append(stat_train[stat_train['Container Unload From Vessel-Actual'] > date2])

print('created training set for summary statistics...')

# create y statistics
stat_route_y = (stat_train.groupby(['Carrier','Original Port Of Loading',
                                    'Final Port Of Discharge'])['y_depart']
                          .mean().reset_index().rename(columns={'y_depart':'mean_all_departed'}))
stat_route_y['mean_all_departed'] = round(stat_route_y['mean_all_departed'], 2)

print('created y statistics')

# schedule statistics
stat_schedule = (stat_train.drop_duplicates(['Original Port Of Loading', 'Final Port Of Discharge', 'Carrier', 'ATD', 'ATA'])
                      .groupby(['Carrier','Original Port Of Loading', 'Final Port Of Discharge',
                                'departure_day'])['route']
                      .agg([np.mean, np.count_nonzero]).reset_index()
                      .rename(columns={'mean':'schedule', 'count_nonzero': 'N'}))
    
stat_schedule = stat_schedule[stat_schedule['N'] > 4].drop('N', axis=1)
stat_schedule['schedule'] = round(stat_schedule['schedule'], 1)


stat_route = (stat_train.drop_duplicates(['Original Port Of Loading', 'Final Port Of Discharge', 'Carrier', 'ATD', 'ATA'])
                  .groupby(['Carrier','Original Port Of Loading',
                            'Final Port Of Discharge'])['route']
                  .agg([np.mean]).reset_index()
                  .rename(columns={'mean':'route_mean'}))
    
stat_route['route_mean'] = round(stat_route['route_mean'], 1)
stat_route = stat_route[stat_route['route_mean'].isna() == False]

print('created schedule and route statistics...')

# pod statistics
stat_pod = (stat_train.groupby(['Final Port Of Discharge', 'customer'])['pod']
                  .agg([np.mean, np.std]).reset_index()
                  .rename(columns={'mean':'pod_mean', 'std': 'pod_std'}))
stat_pod['pod_mean'] = round(stat_pod['pod_mean'], 1)
stat_pod['pod_std'] = round(stat_pod['pod_std'], 1)

print('created pod statistics...')            

# quarter statistics
stat_train = (stat_train.groupby(['Original Port Of Loading',
                                  'Final Port Of Discharge',
                                  'Carrier'])['y_depart'].agg([np.mean, np.std]).reset_index()
                        .merge(stat_train,
                               how='right',
                               on=['Original Port Of Loading',
                                   'Final Port Of Discharge',
                                   'Carrier']))

stat_train['y_zscore'] = (stat_train['y_depart']-stat_train['mean'])/stat_train['std']

stat_quarter = (stat_train.groupby('quarter')['y_zscore']
                          .mean().reset_index()
                          .rename(columns={'y_zscore': 'quarter_z'}))
stat_quarter['quarter_z'] = round(stat_quarter['quarter_z'], 4)

print('created all summary statistics for departed...')


##############################################################################
# get data booking

stat = customer_clean[['Original Port Of Loading', 'Shipper',
                       'Container Unload From Vessel-Actual',
                       'ATD', 'Actual Receipt Date', 'Origin Service']] # for booked date

stat['poo'] = (stat['ATD']-stat['Actual Receipt Date']).dt.days # time goods spent in port of origin after received by shipper

# sanity test
stat = stat[(stat['poo'] > 0) & (stat['poo'] < 30)]

# split into train test for creating statistics:
date = max(stat['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
stat_train = stat[stat['Container Unload From Vessel-Actual'] < date]
# double the observations in the closest year
date2 = max(stat_train['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
stat_train = stat_train.append(stat_train[stat_train['Container Unload From Vessel-Actual'] > date2])

print('created training set for summary statistics...')

# create variables
# port of origin statistics
stat_poo = (stat_train.groupby(['Original Port Of Loading', 'Shipper'])['poo']
                      .agg([np.mean, np.std]).reset_index()
                      .rename(columns={'mean':'poo_mean', 'std': 'poo_std'}))
stat_poo['poo_mean'] = round(stat_poo['poo_mean'], 1)
stat_poo['poo_std'] = round(stat_poo['poo_std'], 1)

print('created poo statistics...')

##############################################################################
# get error metrics

results_booking = pd.read_csv('data/results/booked_test_new.csv').set_index('Unnamed: 0')
results_booking['error'] = results_booking['y_hat']-results_booking['y_book']
results_booking = (results_booking.groupby(['Carrier','Original Port Of Loading','Final Port Of Discharge'])['error']
                                  .quantile(q=[0.2,0.8])
                                  .unstack(level=3)
                                  .reset_index())

print('ready to create model sets...')
