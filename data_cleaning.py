#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:07:36 2018

@author: Florian Krempl

Capstone Project: Predicting Transit time with Machine Learning
Partner Company: DAMCO/Maersk

"""
import os
import numpy as np
import pandas as pd

# set working directory
os.chdir('E:\MIT\Capstone')
exec(open('scripts/functions.py').read())

# list of all column names - to make sure all documents have the same col names
column_names = ['PO Line Uploaded', 'POH Client Date', 'POH Upload Date',
                'Book Date', 'Receipt Date', 'Consolidation Date',
                'ETD', 'ETA', 'ATD', 'ATA', 'Consignee',
                'PO Number', 'Origin Service', 'Destination Service', 'Consignee.1',
                'Carrier', 'VOCC Carrier', 'Carrier SCAC', 'CBL Number',
                'Booking Number', 'Shipper', 'Supplier', 'Buyer', 'Seller',
                'Original Port Of Loading', 'Original Port Of Loading Site',
                'Final Port Of Discharge', 'Final Port Of Discharge Site',
                'Actual Measurement', 'Earliest Receipt Date', 'Expected Receipt Date',
                'Latest Receipt Date', 'Actual Receipt Date',
                'Empty Equipment Dispatch-Actual', 'Gate In Origin-Actual',
                'Container Loaded On Vessel-Actual', 'Consolidation Date.1',
                'Container Unload From Vessel-Actual', 'Gate Out Destination-Actual',
                'Container Empty Return-Actual', 'Equipment Number',
                'Confirmation Date']

# list of customers from which we got data
customers = ['USWA', 'USAD', 'USNI', 'USHO', 'USTA',
             'USDO', 'USCL', 'USHA', 'USHE']
# customer_name = 'USTE' does not work

###############
## load data ##
###############

# define date columns for formatting
date_columns = ['Book Date','Gate In Origin-Actual', 'Actual Receipt Date',
                'Container Unload From Vessel-Actual', 'Expected Receipt Date',
                'Latest Receipt Date', 'ATA', 'ATD', 'ETA', 'ETD']

customer_clean = load_data(customers,
                           folder = '\\data\\Capstone Project data\\',
                           names = column_names 
                           )

# get rid of wrong date rows
customer_clean = customer_clean.drop(customer_clean[customer_clean['Book Date'].str.slice(6,10) > '2500'].index)
customer_clean = customer_clean.drop(customer_clean[customer_clean['Book Date'].str.contains('3016|6186|2528|7340|0006')].index)

for column in date_columns:
    print('start converting column', column, 'to datetime...')
    customer_clean[column] = customer_clean[column].str.slice(0,10).replace('-', '/', regex = True)
    customer_clean[column] = pd.to_datetime(customer_clean[column],
                                            format = '%d/%m/%Y')

print('\n', 'load test dataset...', '\n')

customer_new = load_data(customers,
                         folder = '\\data\\testset\\',
                         names = column_names)

# get rid of wrong date rows
customer_new = customer_new.drop(customer_new[customer_new['Book Date'].str.slice(6,10) > '2500'].index)
customer_new = customer_new.drop(customer_new[customer_new['Book Date'].str.contains('3016|6186|2528|7340|0006')].index)

for column in date_columns:
    print('start converting column', column, 'to datetime...')
    customer_new[column] = customer_new[column].str.slice(0,10).replace('-', '/', regex = True)
    customer_new[column] = pd.to_datetime(customer_new[column],
                                          format = '%m/%d/%Y')

customer_clean = customer_clean.append(customer_new)

##################
# get y columns ##
##################

customer_clean['y_depart'] = (customer_clean['Container Unload From Vessel-Actual']
                              - customer_clean['ATD']).dt.days

customer_clean['y_gate'] = (customer_clean['Container Unload From Vessel-Actual']
                            - customer_clean['Gate In Origin-Actual']).dt.days

customer_clean['y_book'] = (customer_clean['Container Unload From Vessel-Actual']
                            - customer_clean['Book Date']).dt.days

customer_clean['y_receive'] = (customer_clean['Container Unload From Vessel-Actual']
                               - customer_clean['Actual Receipt Date']).dt.days



print('finished calculating y columns')

######################
## load statistics  ##
######################
# this whole next part should be a fancy SQL query on the company database - our data is in csv format - so...

# load chinese new year dates
chinese_holidays = pd.read_csv('data/statistics/chinese_holidays_complete.csv',
                               sep = ',', encoding= 'latin1')
chinese_holidays['date'] = pd.to_datetime(chinese_holidays['date'])

# load us port capacity statistics (this one is a little more complicated than necessary... will change soon)
port_cap = pd.read_csv('data/statistics/summary_ports_us.csv',
                       sep = ',', encoding='latin1')
port_cap = port_cap[['City', 'year', 'Arrival Date', 'cap']]
# get right date format
port_cap['Arrival Date'] = pd.to_datetime(port_cap['Arrival Date'], format = '%d.%m.%Y')
port_cap.index = port_cap['Arrival Date'].dt.dayofyear
# get rolling mean for 3 days for every year
port_cap = (port_cap.groupby(['City', 'year'])['cap']
                    .rolling(3, center = True, min_periods = 1).mean()
                    .reset_index())
port_cap.columns = ['City', 'year', 'doy', 'cap']
# get mean over all years
port_cap = port_cap.groupby(['City', 'doy'])['cap'].mean().reset_index()

# get right string format
port_cap['City'] = port_cap['City'].str.upper()
customer_clean['doy'] = customer_clean['ETA'].dt.dayofyear

###############
## merge dfs ##
###############
print('merging dataframes')
customer_clean = (customer_clean.merge(port_cap, 
                                       left_on = ['Final Port Of Discharge', 'doy'],
                                       right_on = ['City', 'doy'],
                                       how = 'left'))

# create column holiday for chinese holidays
customer_clean['holiday'] = np.where(np.isin(customer_clean['ETD'], chinese_holidays['date']), 1, 0)

print('ready for training')

customer_clean.to_csv('data/customer_clean.csv')
