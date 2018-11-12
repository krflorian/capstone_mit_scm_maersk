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
summary = pd.read_csv('data/valid_routes.csv')
summary = summary.set_index(['Carrier',
                             'Original Port Of Loading',
                             'Final Port Of Discharge'])

valid_routes = summary.index.values.tolist()

# clean data
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
#customer_clean['y']

#schedule miss
customer_clean['schedule_miss'] = customer_clean['ETD'] - customer_clean['ATD']
customer_clean['schedule_miss'] = customer_clean['schedule_miss'].dt.days
#customer_clean['schedule_miss']


# day of week / Quarter
customer_clean['weekday'] = customer_clean['ETD'].dt.weekday_name
customer_clean = customer_clean.join(pd.get_dummies(customer_clean['weekday']))

customer_clean['quarter'] = customer_clean['ETD'].dt.quarter
customer_clean = customer_clean.join(pd.get_dummies(customer_clean['quarter']))

# filter valid routes
customer_clean = customer_clean.set_index(['Carrier',
                                           'Original Port Of Loading',
                                           'Final Port Of Discharge'])

customer_clean = customer_clean.loc[customer_clean.index.isin(valid_routes)]

# get route statistics
customer_clean = customer_clean.join(summary[['median', 'std']])

customer_clean.to_csv("data/customer_clean")

###########################################################
# SUMMARY Statistics - valid routes
#############################################################

summary = customer_clean.groupby(['Carrier',
              'Original Port Of Loading',
              'Final Port Of Discharge'])['y'].agg([np.mean, np.std, np.median, np.count_nonzero])

summary = summary.sort_values("count_nonzero", ascending=0)
n_obs = sum(summary["count_nonzero"])
summary["percent_of_total"] = summary["count_nonzero"] / n_obs
summary["cum_of_total"] = summary["percent_of_total"].cumsum()
summary = summary.loc[summary["cum_of_total"] < 0.9]

valid_routes = summary.index.values.tolist()
summary.to_csv("data/valid_routes.csv")

