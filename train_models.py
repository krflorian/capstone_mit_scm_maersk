# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:07:08 2019

@author: Florian Krempl 

Capstone Project Predict Transit time with machine learning
"""

import os
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

os.chdir('E:\MIT\Capstone')
os.getcwd()

exec(open('scripts/functions.py').read())
exec(open('scripts/setup_stats.py').read())

##########################################################################################
# get relevant variables

# get model departed
customer_clean = customer_clean[customer_clean['ATD'].isna() == False]
customer_clean['departure_day'] = customer_clean.apply(lambda x: x['ATD'].strftime('%A'), axis=1)
customer_clean['late_departure'] = np.where((customer_clean['ATD']-customer_clean['ETD']).dt.days > 0, 1, 0) # if vessel left after scheduled time
customer_clean['quarter'] = customer_clean['ETD'].dt.quarter

print('created departed columns...')
model_route = (customer_clean.merge(stat_pod, on=['Final Port Of Discharge', 'customer'])
                             .merge(stat_route_y, on=['Carrier', 'Original Port Of Loading', 'Final Port Of Discharge'])
                             .merge(stat_route, on=['Carrier', 'Original Port Of Loading', 'Final Port Of Discharge'])
                             .merge(stat_schedule,
                                    on=['Carrier', 'Original Port Of Loading',
                                        'Final Port Of Discharge', 'departure_day'],
                                    how = 'left')
                             .merge(stat_quarter, on=['quarter']))

model_route['schedule'] = np.where(model_route['schedule'].isna(),
                                   model_route['route_mean'],
                                   model_route['schedule'])
print('merged departed statistics...')
model_route = model_route[(model_route['y_depart'] > 0) & (model_route['y_depart'] < 40)]
model_route_ready = model_route[['quarter_z', 'pod_mean', 'pod_std', 'mean_all_departed',
                           'cap', 'late_departure', 'schedule', 'y_depart',
                           'Container Unload From Vessel-Actual']]
model_route_ready = model_route_ready.dropna()
print('model route ready!')


model_departed, features = get_features_departed(customer_clean, model_training=True)
model_departed = model_departed.dropna(subset = features)
model_departed_ready = model_departed[features]
print('model received ready...')


###############################################################################################
# get model received

model_received, features = get_features_received(customer_clean, model_training=1)
model_received = model_received.dropna(subset = features)
model_received_ready = model_received[features]
print('model received ready...')

###############################################################################################
# get variables booked

model_booked, features = get_features_booking(customer_clean, model_training=1)
model_booked = model_booked.dropna(subset = features)
model_booked_ready = model_booked[features]
print('model booked ready...')

##########################################################################################
##########################################################################################

#X_route = ['quarter_z', 'pod_mean', 'pod_std', 'mean_all_departed', 'cap', 'late_departure', 'schedule']
model = model_received_ready
model = model.rename(columns={'y_depart':'y', 'y_book':'y', 'y_receive':'y'})

date = max(model['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
train = model[model['Container Unload From Vessel-Actual'] < date].drop('Container Unload From Vessel-Actual', axis=1)
test = model[model['Container Unload From Vessel-Actual'] >= date].drop('Container Unload From Vessel-Actual', axis=1)

metrics_total_test = pd.DataFrame({'trees': [],
                            'depth': [],
                            'MAPE': [],
                            'MAE': []})
metrics_total_train = pd.DataFrame({'trees': [],
                            'depth': [],
                            'MAPE': [],
                            'MAE': []})

X_train = train.drop('y', axis=1)
y_train = train['y']
X_test = test.drop('y', axis=1)
y_test = test['y']

#########################################################################################


# random forest
for depth in [15,16,17]: 
    for tree in [500]:
        rf = randomforest(max_depth = depth, trees = tree, features = 'sqrt')
        rf.fit(X_train, y_train)

        y_hat = rf.predict(X_test)
        m = print_metrics(y_hat, y_test)
        metrics_test = pd.DataFrame({'trees': [tree],
                                     'depth': [depth],
                                     'MAPE': [m['MAPE'][0]],
                                     'MAE': [m['MAE'][0]]})
        metrics_total_test = metrics_total_test.append(metrics_test)
        
        y_hat_train = rf.predict(X_train)
        m = print_metrics(y_hat_train, y_train)
        metrics_train = pd.DataFrame({'trees': [tree],
                                      'depth': [depth],
                                      'MAPE': [m['MAPE'][0]],
                                      'MAE': [m['MAE'][0]]})
        metrics_total_train = metrics_total_train.append(metrics_train)
        print('\n', metrics_test)
        print('\n', 'finished tree: ', tree, '\n', 'depth: ', depth)

metrics_total_train
metrics_total_test.sort_values(['MAPE', 'MAE']).head(5)

metrics_total_test.to_csv('data/results/randomforest_test_received_3.csv')
metrics_total_train.to_csv('data/results/randomforest_train_received_3.csv')

feature_importance = (pd.DataFrame(test.drop('y',axis=1).columns,
                                   rf.feature_importances_)
                        .reset_index()
                        .sort_values('index', ascending=0))
feature_importance

# save output

output = model_received.ix[list(test.index.values)]
output['y_hat'] = y_hat
output.to_csv('data/results/route_received_new.csv')
filename = 'data/models/randomforest_received_1_4.joblib'
joblib.dump(rf, filename) 

##################################################################################################

# neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_train_norm = pd.DataFrame(X_train_norm)

nn = neural_net(X_train_norm, neurons=50)

history = nn.fit(X_train_norm, y_train, validation_split = 0.2,
                 epochs = 25, batch_size = 500)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm)

y_hat = nn.predict(X_test_norm)

m = print_metrics(y_hat, y_test)
m.to_csv('data/results/neuralnetwork_test_booked.csv')



##################################################################################################

# linear regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print('fitted linear model...')

coef = pd.DataFrame({'names': X_train.columns, 'coeficients': regr.coef_})
y_hat = regr.predict(X_test)

print(print_metrics(y_hat, y_test))
print(coef)

#############################################################################################################################
# base case
model = customer_clean[(customer_clean['y_book'] > 0) & (customer_clean['y_book'] < 100)]
model = model.rename(columns={'y_receive':'y'})

date = max(model['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
train = model[model['Container Unload From Vessel-Actual'] < date].drop('Container Unload From Vessel-Actual', axis=1)
test = model[model['Container Unload From Vessel-Actual'] >= date].drop('Container Unload From Vessel-Actual', axis=1)

test_base = (model.groupby(['Carrier','Original Port Of Loading',
                            'Final Port Of Discharge'])['y']
                  .mean().reset_index().rename(columns={'y':'mean'})
                  .merge(model,
                         on=['Carrier','Original Port Of Loading', 'Final Port Of Discharge'],
                         how='right'))
    
test_base['mean'] = round(test_base['mean'], 2)

test_base = test_base[np.isnan(test_base['y_depart']) == False]
test_base = test_base[np.isnan(test_base['mean']) == False]

print_metrics(test_base['mean'], test_base['y_depart'])
test_base.columns

################################################################################################################################
