#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:36:19 2018

@author: Florian Krempl

Capstone - DAMCO
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()


customer1 = pd.read_csv('data/data1.csv')
customer2 = pd.read_csv('data/data2.csv')
customer1 = customer1.loc[:, customer1.columns != 'Consolidation Date.1'].copy()
customer2 = customer2.loc[:, customer2.columns != 'Consolidation Date.1'].copy()

customer2['ConsigneeName'] = customer1.ConsigneeName[0] + '2'
customer2['Consignee'] = customer1.Consignee[0] + '2'

customer = customer1.append(customer2)

#customer.head()

"""
    SETUP
    
"""
# transform to date type and clean missing values
date_columns = ['ATA', 'ATD','Actual Receipt Date',
                'Book Date','Confirmation Date', 
                'Consolidation Date', 'Container Empty Return-Actual', 'Container Loaded On Vessel-Actual',
                'Container Unload From Vessel-Actual', 'ETA',
                'ETD', 'Earliest Receipt Date', 'Empty Equipment Dispatch-Actual',
                'Expected Receipt Date', 'Gate In Origin-Actual', 'Gate Out Destination-Actual', 'Latest Receipt Date',
                'PO Line Uploaded', 'POH Client Date', 'POH Upload Date', 'Receipt Date']

#date_columns += list(customer.columns[3:10])
#date_columns += list(customer.columns[26:36])
date_columns

category_columns = ['Carrier','Shipper', 'ConsigneeName',
                    'Original Port Of Loading', 'Final Port Of Discharge',]
#category_columns += list(customer.columns[[24, 22, 20, 15]])

customer_clean = customer.dropna(subset= ['ATA', 'ATD', 'ETD']).reset_index().copy()

for column in date_columns:
    customer_clean[column] = pd.to_datetime(customer_clean[column])
    print(['finished converting column', column, 'to date'])

for column in category_columns:
    customer_clean[column] = customer_clean[column].astype('category')
    print(['finished converting column', column, 'to category'])
    
customer_clean['schedule_miss'] = customer_clean['ATD'] - customer_clean['ETD']
customer_clean['schedule_miss'] = customer_clean['schedule_miss'].dt.days



print(customer_clean.dtypes)
print(customer_clean.columns)


"""
Test 1 ATD to ATA 
"""


# setup

input_columns = ['Carrier',
                 'Shipper',
                 'Original Port Of Loading',
                 'Final Port Of Discharge',
                 'ConsigneeName',
                 'ETD',
                 'schedule_miss'
                 ]

y_column = ['ATD', 'ATA']

customer_clean = customer_clean.dropna(subset=input_columns)

x = customer_clean[input_columns]


x['weekday'] = x['ETD'].dt.weekday_name.astype('category') #create column with weekdays of schedule of departure
x = x.loc[:, x.columns != 'ETD']  #delete column ETD from dataframe x
x.head()
x.dtypes

tfweekday = preprocessing.LabelEncoder()
tfweekday.fit(x['weekday'])

tfshipper = preprocessing.LabelEncoder()
tfshipper.fit(x['Shipper'])

tfconsignee = preprocessing.LabelEncoder()
tfconsignee.fit(x['ConsigneeName'])

tfcarrier = preprocessing.LabelEncoder()
tfcarrier.fit(x['Carrier'])

tforigin = preprocessing.LabelEncoder()
tforigin.fit(x['Original Port Of Loading'])

tfdest = preprocessing.LabelEncoder()
tfdest.fit(x['Final Port Of Discharge'])

#tfcarrier = preprocessing.LabelEncoder()
#tfcarrier.fit(x['Carrier'])
#list(tfcarrier.classes_)
#x['Carrier']
#tfcarrier

list(tfdest.classes_)
list(tforigin.classes_)
list(tfshipper.classes_)
list(tfweekday.classes_)
list(tfcarrier.classes_)
list(tfconsignee.classes_)

X_new = pd.DataFrame()
X_new['shipper'] = tfshipper.transform(x['Shipper'])
X_new['consignee'] = tfconsignee.transform(x['ConsigneeName'])
X_new['carrier'] = tfcarrier.transform(x['Carrier'])
X_new['origin'] = tforigin.transform(x['Original Port Of Loading'])
X_new['destination'] = tfdest.transform(x['Final Port Of Discharge'])
X_new['weekday'] = tfweekday.transform(x['weekday'])
X_new['schedule_miss'] = customer_clean['schedule_miss']
X_new.columns

y = customer_clean[y_column[1]] - customer_clean[y_column[0]]
y = y.dt.days
y.head()
y.dtypes

print(x.shape)
print(x.dtypes)
print(y.shape)
print(y.dtypes)

# write csv with train data
writer = pd.ExcelWriter('train.xlsx')
X_new.to_excel(writer, 'sheet1')
writer.save()

writer = pd.ExcelWriter('test.xlsx')
X_new.to_excel(writer, 'sheet1')
writer.save()



"""
    see milestones - not in real skript
    
"""



#calculations
#for column in date_columns:
#    print([column, (customer[column][5] - customer['Earliest Receipt Date'][5]).days])
#
#customer['ATA'][0] - customer['Receipt Date'][0]


# get milestone order
dates = customer_clean[date_columns]
milestone = dates[15:17].transpose().reset_index().copy()
milestone.columns = ['names', 'row1', 'row2']

milestone = milestone.sort_values(by=['row1'])

milestone
#write excel
writer = pd.ExcelWriter('milestones2.xlsx')
milestone.to_excel(writer, 'sheet1')
writer.save()


milestone



