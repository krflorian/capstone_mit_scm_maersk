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

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

customer_clean = pd.DataFrame({'Carrier': [], 'ATA': [], 'ATD': [], 'ETD': [],
                              'Original Port Of Loading': [],
                              'Final Port Of Discharge': [],
                              'Original Port Of Loading Site': [],
                              'Final Port Of Discharge Site': []})

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

customers = ['USWA', 'USAD', 'USNI', 'USHO', 'USTA',
             'USDO', 'USCL']
# customer_name = 'USHA' and 'USHE', 'USTE' do not work

for customer_name in customers:
    print("start to load", customer_name, "...") 
    customer = pd.read_csv('data/Capstone Project data/' + customer_name +'.csv')
    print("loaded", customer_name, "starting to clean the dataframe")
    
    if customer.columns[0] == 'Unnamed: 0':
        customer = customer.drop(columns='Unnamed: 0')
        print('droped unnecessary index column')
    customer.columns = column_names

    # get real carrier
    customer["Carrier"] = np.where(customer['VOCC Carrier'].isna,
            customer["CBL Number"],    #"Carrier SCAC"
            customer["VOCC Carrier"])
    
    # get rid of missing data - select right columns
    customer = (customer.loc[customer['Final Port Of Discharge Site'] == 'UNITED STATES']
                        .loc[customer['ATA'].notna()]
                        .loc[customer['ATD'].notna()]
                        .loc[customer['ETD'].notna()]
                        .loc[:,['Carrier', 'ATA', 'ATD', 'ETD',
                                'Original Port Of Loading',
                                'Final Port Of Discharge',
                                'Original Port Of Loading Site',
                                'Final Port Of Discharge Site']]
                        .drop_duplicates(subset=['Carrier','ATD',
                                                 'Original Port Of Loading',
                                                 'Final Port Of Discharge']))
    customer['customer'] = customer_name
    # append to customer_clean
    print("appending", customer_name, "to customer_clean", "\n") 
    customer_clean = customer_clean.append(customer)

# get date to right format
date_columns = ['ATA', 'ATD', 'ETD']
customer_clean[date_columns] = customer_clean[date_columns].replace('-', '/', regex = True)

for column in date_columns:
    customer_clean[column] = pd.to_datetime(customer_clean[column], format =  '%d/%m/%Y')
    print(['finished converting column', column, 'to date'])

#get y column
customer_clean['y'] = customer_clean['ATA'] -  customer_clean['ATD']
customer_clean['y'] = customer_clean['y'].dt.days
#customer_clean['y']

#schedule miss
customer_clean['schedule_miss'] = customer_clean['ETD'] - customer_clean['ATD']
customer_clean['schedule_miss'] = customer_clean['schedule_miss'].dt.days
#customer_clean['schedule_miss']


###########################################################
# SUMMARY Statistics - valid routes
#############################################################

# get summary exact
summary = customer_clean.groupby(['customer', 'Carrier',
                                  'Original Port Of Loading','Final Port Of Discharge']
        )['y'].agg([np.mean, np.std, np.median, np.count_nonzero])

summary = summary[summary['count_nonzero'] > 50]
summary.to_csv("data/summary_route_customer.csv")
summary = summary.reset_index()
# get route statistics
customer_clean = customer_clean.merge(summary,
                                      on = ['customer', 'Carrier',
                                            'Original Port Of Loading',
                                            'Final Port Of Discharge'],
                                      how = 'left')

# get summary statistic w/o customer
summary = customer_clean.groupby(['Carrier',
              'Original Port Of Loading',
              'Final Port Of Discharge'])['y'].agg([np.mean, np.std, np.median, np.count_nonzero])

summary = summary[summary['count_nonzero'] > 50]
summary.to_csv("data/summary_route_carrier.csv")
summary = summary.reset_index()

customer = (customer_clean.loc[np.isnan(customer_clean['mean'])]
                          .drop(['mean', 'median', 'count_nonzero', 'std'], axis = 1)
                          .merge(summary,
                                 on = ['Carrier','Original Port Of Loading',
                                       'Final Port Of Discharge'],
                                 how = 'left'))

customer_clean = customer_clean.append(customer)
customer_clean = customer_clean.loc[customer_clean['mean'].notna()]

customer_clean = customer_clean.drop(['mean', 'count_nonzero'], axis=1)
customer_clean.to_csv("data/customer_clean_route")
