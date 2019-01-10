#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:13:04 2018

@author: Florian Krempl
"""

from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import os
import datetime

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

exec(open('scripts/capstone_datapipeline_cleaning').read())

customer_clean['late arrival'] = (customer_clean['ETA']-customer_clean['ATA']).dt.days
customer_clean['actual'] = (customer_clean['ATA']-customer_clean['Gate In Origin-Actual']).dt.days
customer_clean = customer_clean.dropna()


X = customer_clean[['actual', 'late arrival','consolidation', 'std_po', 'median_po', 'holiday',
                    'std_pd', 'median_pd', 'cap',
                    'mean_schedule', 'std_route', 1, 2, 3,
                    'USAD', 'USDO', 'USHA', 'USHE', 'USHO', 'USTA']]

y = customer_clean['y']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1992)


##############################
## test with random forest  ##
##############################

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

regr.fit(X_train, y_train)

y_hat = regr.predict(X_test) #.drop('y_hat', axis=1)

print_metrics(y_hat, y_test)

feature_importance = (pd.DataFrame(X_train.columns,
                                  regr.feature_importances_)
                        .reset_index()
                        .sort_values('index', ascending=0))
print(feature_importance)

filename = 'data/models/randomforest_pod_1_1.joblib'

from sklearn.externals import joblib
joblib.dump(regr, filename) 
regr = joblib.load('data/models/randomforest_pod_1_1.joblib')

####################################
##  multiple linear regression
####################################


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(n_jobs = -1)

lin_reg.fit(X_train, y_train)

y_hat = lin_reg.predict(X_test)

print_metrics(y_hat, y_test)

####################################
##  boosted trees
####################################


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
boost_regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                               n_estimators=300, random_state=rng)

boost_regr.fit(X_train, y_train)

boost_regr.fit(X_train, y_train)
y_hat = boost_regr.predict(X_test)

test_data = pd.DataFrame(data = {'y_hat': y_hat,
                                 'y': y_test})

print_metrics(test_data["y_hat"], test_data["y"])


