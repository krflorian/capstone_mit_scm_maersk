#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:13:04 2018

@author: drx
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
import os
import datetime

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

customer_clean = pd.read_csv('data/customer_clean_pod_USDO.csv')

X = customer_clean[['late arrival','mean', 'amax', 'std carrier', 'median']]

y = customer_clean['y']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1992)


############################################
##   train RANDOM FOREST
############################################

#regr = joblib.load('data/models/randomforest_11_11.joblib')
regr = RandomForestRegressor(n_estimators = 10,
                             criterion = 'mae',
                             random_state = 1992,
                             max_depth = 15,
                             max_leaf_nodes = None,
                             n_jobs = -1,
                             verbose = 2
                             )

print(datetime.datetime.now())
regr.fit(X_train, y_train)
print(datetime.datetime.now())

###########################################
##  get model metrics
###########################################

feature_importance = pd.DataFrame(X_train.columns,
                                  regr.feature_importances_)
print(feature_importance)

y_hat = regr.predict(X_test)

print(regr.score(X_test, y_test))

test_data = pd.DataFrame(data = {'y_hat': y_hat,
                                 'y': y_test})

def print_metrics(y_hat, y_test):
    output = print('features: ', feature_importance, "\n",
                   'MAE: ', round(np.mean(abs(y_hat-y_test)), 2), "\n",
                   'MAPE: ', round(np.mean(abs(y_hat-y_test)/y_test), 4), "\n",
                   'RMSE: ', round(np.sqrt(np.mean((y_test - y_hat)**2)), 2), "\n"
                   ' R2: ', round(metrics.r2_score(y_test, y_hat),2)
                   )
    return output

print_metrics(test_data["y_hat"], test_data["y_test"])

test_data['y_hat_round'] = round(test_data['y_hat'],0)

sum(test_data['y_hat_round'] == test_data['y_test']) / len(test_data)
sum(test_data['y_hat_round']+1 > test_data['y_test']) and sum(test_data['y_hat_round']-1 < test_data['y_test']) / len(test_data)

report = test_data.join(customer_clean, how='left')
report.to_csv("data/report_pod_USDO.csv")

#np.mean(abs(y_hat-y_test)/y_test)

###################################################
##      save model
###################################################

now = datetime.datetime.now()

filename = 'data/models/randomforest_pod' + ''.join(['_', str(now.month), '_', str(now.day)]) +'.joblib'

joblib.dump(regr, filename) 

