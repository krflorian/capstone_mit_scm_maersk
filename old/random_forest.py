#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:59:16 2018

@author: Florian Krempl
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import os


os.chdir('E:\MIT\Capstone')
os.getcwd()

pd.set_option('display.expand_frame_repr', False)
seed = 1992
np.random.seed(seed)

# execute datapipeline script - load all customer data and functions
exec(open('scripts/capstone_datapipeline_cleaning').read())
exec(open('scripts/functions').read())

date = max(customer_clean['ATA'])- datetime.timedelta(days=14)
############################
# 1st milestone           ##
# train from booking date ##
############################
    
# set up training and test set
# get right columns for 1st milestone
X = customer_clean[['time_to_port', 'consolidation', 'std_po', 'median_po', 'holiday',
                    'std_pd', 'median_pd', 'cap',
                    'mean_schedule', 'std_route', 1, 2, 3,
                    'USAD', 'USDO', 'USHA', 'USHE', 'USHO', 'USTA', 'y_book']]
X = X.dropna()
X = X[(X['y_book'] > 0)]
y = X['y_book']
X = X.drop('y_book', axis = 1)

# split data set in training and test set
#X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.3, random_state=1992)

# either load model from saved model or new from function
#rf_book = joblib.load('data/models/randomforest_book_1_2.joblib')
rf_book = randomforest()
# train model
rf_book.fit(X, y)

# predict new values on test set
#y_hat_book = rf_book.predict(X_test)
y_hat_book = rf_book.predict(X)
print_metrics(y_hat_book, y)

# save model for serialization
filename = 'data/models/randomforest_book_1_2.joblib'
joblib.dump(rf_book, filename) 

#########################
## 2nd milestone       ##
## train from gate in  ## 
#########################

# get right columns for 2nd milestone
X = customer_clean[['consolidation', 'std_po', 'median_po', 'holiday',
                    'std_pd', 'median_pd', 'cap',
                    'mean_schedule', 'std_route', 1, 2, 3,
                    'USAD', 'USDO', 'USHA', 'USHE', 'USHO', 'USTA', 'y_gate']]
X = X.dropna()
X = X[(X['y_gate'] > 0)]
y = X['y_gate']
X = X.drop('y_gate', axis = 1)

#X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.3, random_state=1992)

rf_gate = joblib.load('data/models/randomforest_gate_1_2.joblib')
rf_gate = randomforest()
rf_gate.fit(X, y)

y_hat_gate = rf_gate.predict(X)
print_metrics(y_hat_gate, y)

# save model
filename = 'data/models/randomforest_gate_1_2.joblib'
joblib.dump(rf_gate, filename) 

##########################
## 3rd milestone        ##
## train from received  ##
##########################

# get right columns for 3rd milestone
X = customer_clean[['consolidation', 'std_po', 'median_po', 'holiday',
                    'std_pd', 'median_pd', 'cap',
                    'mean_schedule', 'std_route', 1, 2, 3,
                    'USAD', 'USDO', 'USHA', 'USHE', 'USHO', 'USTA', 'y_receive']]
X = X.dropna()
X = X[(X['y_receive'] > 0)]
y = X['y_receive']
X = X.drop('y_receive', axis = 1)

#X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.3, random_state=1992)

#rf_received = joblib.load('data/models/randomforest_received_1_2.joblib')
rf_received = randomforest()
rf_received.fit(X, y)

y_hat_received = rf_received.predict(X)
print_metrics(y_hat_received, y)

# save model
filename = 'data/models/randomforest_received_1_2.joblib'
joblib.dump(rf_received, filename) 

##########################
## 4th milestone        ##
## train from departed  ##
##########################

X, y = get_x_departed(customer_clean)

#X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.3, random_state=1992)

rf_departed = randomforest(max_depth = 20)
rf_departed.fit(X, y)



y_hat_departed = rf_departed.predict(X)
print_metrics(y_hat_departed, y)

# save model
filename = 'data/models/randomforest_departed_1_2.joblib'
joblib.dump(rf_departed, filename) 


############################################
## test module - with feature importance  ##
############################################
rf_book = joblib.load('data/models/randomforest_gate_1_2.joblib')
rf_departed = joblib.load('data/models/randomforest_departed_1_2.joblib')
rf_received = joblib.load('data/models/randomforest_received_1_2.joblib')
rf_gate = joblib.load('data/models/randomforest_gate_1_2.joblib')


# change to whatever values to test
y_hat = rf_book.predict(X_test)
y_hat = rf_departed.predict(X)

print_metrics(y_hat, y)

feature_importance = (pd.DataFrame(X_train.columns,
                                  rf_received.feature_importances_)
                        .reset_index()
                        .sort_values('index', ascending=0))
print(feature_importance)


##############################################################################

#date = max(customer_clean['ATA'])- datetime.timedelta(days=60)
customer_clean_latest = customer_clean[(customer_clean['ATA']<date) & (customer_clean['ATA'] > max(customer_clean['ATA']) - datetime.timedelta(days=100))]
customer_new = customer_clean.append(customer_clean_latest)

X_train, y_train = get_x_departed(customer_new[customer_clean['ATA']<date])
X_test, y_test = get_x_departed(customer_new[customer_clean['ATA']>=date])

metrics_total = pd.DataFrame({'trees': [],
                            'depth': [],
                            'MAPE': [],
                            'MAE': []})

for depth in range(5,50,2):
    for tree in range(10,500,10):
        rf_departed = randomforest(max_depth = 15, trees = 100)
        rf_departed.fit(X_train, y_train)
        
        y_hat_departed = rf_departed.predict(X_test)
        m = print_metrics(y_hat_departed, y_test)
        metrics_test = pd.DataFrame({'trees': [tree],
                                'depth': [depth],
                                'MAPE': [m['MAPE'][0]],
                                'MAE': [m['MAE'][0]]})
        metrics_total_test = metrics_total.append(metrics_test)
        
        y_hat_departed = rf_departed.predict(X_train)
        m = print_metrics(y_hat_departed, y_train)
        metrics_train = pd.DataFrame({'trees': [tree],
                                'depth': [depth],
                                'MAPE': [m['MAPE'][0]],
                                'MAE': [m['MAE'][0]]})
        metrics_total_train = metrics_total.append(metrics_train)


customer_clean.to_csv("data/customer_clean.csv")

