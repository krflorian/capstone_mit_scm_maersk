# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:03:28 2019

@author: kr_fl
"""

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

#####################
## create rf model ##
#####################

def randomforest():
    regr = RandomForestRegressor(n_estimators = 10,
                             criterion = 'mse',
                             random_state = 1992,
                             max_depth = None,
                             max_leaf_nodes = None,
                             min_samples_split = 2,
                             min_samples_leaf = 10,
                             n_jobs = -1,
                             verbose = 100
                             )
    return regr
    
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
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1992)

# either load model from saved model or new from function
#rf_book = joblib.load('data/models/randomforest_book_1_1.joblib')
rf_book = randomforest()
# train model
rf_book.fit(X_train, y_train)

# predict new values on test set
y_hat_book = rf_book.predict(X_test)

# save model for serialization
filename = 'data/models/randomforest_book_1_1.joblib'
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

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1992)

#rf_book = joblib.load('data/models/randomforest_book_1_1.joblib')
rf_gate = randomforest()
rf_gate.fit(X_train, y_train)

y_hat_gate = rf_gate.predict(X_test)
print_metrics(y_hat_gate, y_test)

# save model
filename = 'data/models/randomforest_gate_1_1.joblib'
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

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1992)

#rf_book = joblib.load('data/models/randomforest_book_1_1.joblib')
rf_received = randomforest()
rf_received.fit(X_train, y_train)

y_hat_received = rf_received.predict(X_test)
print_metrics(y_hat_received, y_test)

# save model
filename = 'data/models/randomforest_received_1_1.joblib'
joblib.dump(rf_received, filename) 

##########################
## 4th milestone        ##
## train from departed  ##
##########################

X = customer_clean[['late departure','consolidation', 'std_po', 'median_po', 'holiday',
                    'std_pd', 'median_pd', 'cap',
                    'mean_schedule', 'std_route', 1, 2, 3,
                    'USAD', 'USDO', 'USHA', 'USHE', 'USHO', 'USTA', 'y_depart']]
X = X.dropna()
X = X[(X['y_depart'] > 0)]
y = X['y_depart']
X = X.drop('y_depart', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1992)

#rf_book = joblib.load('data/models/randomforest_book_1_1.joblib')
rf_departed = randomforest()
rf_departed.fit(X_train, y_train)

y_hat_departed = rf_departed.predict(X_test)
print_metrics(y_hat_departed, y_test)

# save model
filename = 'data/models/randomforest_departed_1_1.joblib'
joblib.dump(rf_departed, filename) 



############################################
## test module - with feature importance  ##
############################################

# change to whatever values to test
y_hat = rf_book.predict(X_test)

print_metrics(y_hat_book, y_test)

feature_importance = (pd.DataFrame(X_train.columns,
                                  rf_received.feature_importances_)
                        .reset_index()
                        .sort_values('index', ascending=0))
print(feature_importance)

