#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:36:19 2018

@author: Florian Krempl

Capstone - DAMCO
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

customer = pd.read_csv('data/data2.csv')

customer_clean = (customer.loc[customer['ATA'].notna()]
                          .loc[customer['ATD'].notna()]
                          .loc[:,['Carrier', 'ATA', 'ATD', 'ETD',
                                  'Original Port Of Loading',
                                  'Final Port Of Discharge']])
# get date to right format
date_columns = ['ATA', 'ATD', 'ETD']

for column in date_columns:
    customer_clean[column] = pd.to_datetime(customer_clean[column])
    print(['finished converting column', column, 'to date'])

#get y column
customer_clean['y'] = customer_clean['ATA'] -  customer_clean['ATD']
customer_clean['y'] = customer_clean['y'].dt.days
customer_clean['y']

#schedule miss
customer_clean['schedule_miss'] = customer_clean['ETD'] - customer_clean['ATD']
customer_clean['schedule_miss'] = customer_clean['schedule_miss'].dt.days
customer_clean['schedule_miss']

# carrier statistics
customer_clean['route_std'] = customer_clean.groupby(['Carrier',
              'Original Port Of Loading',
              'Final Port Of Discharge'])['y'].transform(np.std)

customer_clean['route_med'] = customer_clean.groupby(['Carrier',
              'Original Port Of Loading',
              'Final Port Of Discharge'])['y'].transform(np.median)

# day of week / Quartal
customer_clean['weekday'] = customer_clean['ETD'].dt.weekday_name
customer_clean = customer_clean.join(pd.get_dummies(customer_clean['weekday']))

customer_clean['quarter'] = customer_clean['ETD'].dt.quarter
customer_clean = customer_clean.join(pd.get_dummies(customer_clean['quarter']))










