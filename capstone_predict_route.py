#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:02:58 2018

@author: Florian Krempl
"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor 

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

#load model
regr = joblib.load('data/models/randomforest_11_11.joblib')

# load summary statistics
summary = pd.read_csv('data/valid_routes.csv')
summary = summary.set_index(['Carrier',
                             'Original Port Of Loading',
                             'Final Port Of Discharge'])
# what to predict?
#('COSU', 'YANTIAN', 'ROTTERDAM', 02/25/2017, 02/25/2017)

X = pd.DataFrame({'Carrier': ['COSU'],
                 'Original Port Of Loading': ['YANTIAN'],
                 'Final Port Of Discharge': ['ROTTERDAM'],
                 'ETD': ["02/25/2017"], 
                 'ATD': ["02/25/2017"]})

date_columns = ['ETD', 'ATD']

for column in date_columns:
    X[column] = pd.to_datetime(X[column])
    print(['finished converting column', column, 'to date'])

X = X.set_index(['Carrier',
                 'Original Port Of Loading',
                 'Final Port Of Discharge'])

# get statistics
X = X.join(summary[['median', 'std']])

# get quarter
X['quarter'] = X['ETD'].dt.quarter
X = X.join(pd.get_dummies(X['quarter']))
#get missing columns




# get schedule miss
X['schedule_miss'] = X['ETD'] - X['ATD']
X['schedule_miss'] = X['schedule_miss'].dt.days

# get final columns
X = X['schedule_miss', 1,2,3,'median', 'std']

# predict
regr.predict(X_test)




