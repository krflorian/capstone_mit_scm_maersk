#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:11:21 2018

@author: drx
"""


import os
import numpy as np
import pandas as pd


os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

customer = pd.read_csv('data/Capstone Project data/USDO - Oct15-Sept18.csv')

# clean data
# get rid of missing data - select right columns

customer["Carrier"] = np.where(customer['VOCC Carrier'].isna,
        customer["CBL Number"],    #"Carrier SCAC"
        customer["VOCC Carrier"])

customer_clean = (customer.loc[customer['Container Unload From Vessel-Actual'].notna()]
                          .loc[customer['ATA'].notna()]
                          .loc[customer['ETA'].notna()]
                          .loc[:,['Carrier', 'Final Port Of Discharge',
                                  'Final Port Of Discharge Site',
                                  'Container Unload From Vessel-Actual',
                                  'ATA', 'ETA', 'Equipment Number']])

customer_clean = customer_clean.loc[customer_clean['Final Port Of Discharge Site'] == 'UNITED STATES']

# drop all single shipments - keep only full containers
customer_clean = customer_clean.drop_duplicates(subset = 'Equipment Number', keep = 'first')

# get date to right format

date_columns = ['Container Unload From Vessel-Actual', 'ATA', 'ETA']

customer_clean[date_columns] = customer_clean[date_columns].replace('-', '/', regex = True)

for column in date_columns:
    customer_clean[column] = pd.to_datetime(customer_clean[column], format =  '%d/%m/%Y')
    print(['finished converting column', column, 'to date'])

# get features
customer_clean['y'] = customer_clean['Container Unload From Vessel-Actual']-customer_clean['ATA']
customer_clean['y'] = customer_clean['y'].dt.days

customer_clean['late arrival'] = customer_clean['ATA']-customer_clean['ETA']
customer_clean['late arrival'] = customer_clean['late arrival'].dt.days

# get geography

west_coast = ['Los Angeles', 'Long Beach', 'San Pedro',
              'Oakland', 'Seattle', 'Tacoma']
east_coast = ['Savannah', 'Newark', 'Houston', 'New York', 
              'Norfolk']

customer_clean['coast'] = np.where(np.isin(customer_clean['Final Port Of Discharge'], west_coast), 'west coast', 'east coast')

# get port statistics

customer_clean['Final Port Of Discharge'] = customer_clean['Final Port Of Discharge'].str.title()
customer_clean['week'] = customer_clean['ATA'].dt.week

port = pd.read_csv("data/port_summary_us.csv")

port_stat = (port.groupby(['City', 'week'])['cap'].agg([np.mean, np.max])
                 .reset_index())

port_stat_coast = (port.groupby(['coast', 'week'])['cap'].agg([np.mean, np.max])
                       .reset_index())

customer_clean = customer_clean.merge(port_stat, 
                                      how = 'left',
                                      left_on = ['week', 'Final Port Of Discharge'],
                                      right_on = ['week', 'City'])

port_stat_coast.columns = ['coast', 'week', 'mean_replace', 'amax_replace']

customer_clean = customer_clean.merge(port_stat_coast,
                                      how = 'left',
                                      on = ['coast', 'week'])

customer_clean['mean'] = np.where(customer_clean['mean'].isna(),
                                  customer_clean['mean_replace'],
                                  customer_clean['mean'])
customer_clean['amax'] = np.where(customer_clean['amax'].isna(),
                                  customer_clean['amax_replace'],
                                  customer_clean['amax'])
customer_clean = customer_clean.drop(['amax_replace', 'mean_replace'], axis=1)

# get carrier statistics

summary = (customer_clean.groupby(['Carrier',
                                  'Final Port Of Discharge'])['y']
                         .agg([np.mean, np.std, np.median, np.count_nonzero]))

summary = summary.reset_index()

summary.columns = ['Carrier', 'Final Port Of Discharge',
                   'mean carrier', 'std carrier', 'median',
                   'count_nonzero']

customer_clean = customer_clean.merge(summary,
                                      on = ['Carrier',
                                            'Final Port Of Discharge'],
                                      how = 'inner')

customer_clean = customer_clean[customer_clean['count_nonzero'] > 100]
customer_clean = customer_clean.drop(['mean carrier',
                                      'count_nonzero',
                                      'Equipment Number'], axis=1)

# save csv

customer_clean.to_csv("data/customer_clean_pod_USDO.csv")
summary.to_csv("data/summary_pod_USDO.csv")

