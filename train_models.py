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
model_departed, features = get_features_departed(customer_clean, model_training=True)
model_departed = model_departed.dropna(subset = features)
model_departed_ready = model_departed[features]
print('model departed ready...')

###############################################################################################
# get model received

model_received, features = get_features_received(customer_clean, model_training=1)
model_received = model_received.dropna(subset = features)
model_received_ready = model_received[features]
print('model received ready...')

###############################################################################################
# get model gate_in

model_gate, features = get_features_gate(customer_clean, model_training=1)
model_gate = model_gate.dropna(subset = features)
model_gate_ready = model_gate[features]
print('model gate_in ready...')

###############################################################################################
# get variables booked

model_booked, features = get_features_booking(customer_clean, model_training=1)
model_booked = model_booked.dropna(subset = features)
model_booked_ready = model_booked[features]
print('model booked ready...')

##########################################################################################
##########################################################################################

#X_route = ['quarter_z', 'pod_mean', 'pod_std', 'mean_all_departed', 'cap', 'late_departure', 'schedule']
model = model_gate_ready
model = model.rename(columns={'y_depart':'y', 'y_book':'y',
                              'y_receive':'y', 'y_gate':'y'})

date = max(model['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
train = model[model['Container Unload From Vessel-Actual'] < date].drop('Container Unload From Vessel-Actual', axis=1)
test = model[model['Container Unload From Vessel-Actual'] >= date].drop('Container Unload From Vessel-Actual', axis=1)


filename = 'data/models/randomforest_' + 'gate' +'_final.joblib'
rf = joblib.load(filename)

y_hat = rf.predict(test.drop('y', axis=1))
output = model_gate.ix[list(test.index.values)]
output['y_hat'] = y_hat
output.columns
output.to_csv('data/results/model_gate_new.csv')






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
for depth in [9]: 
    for tree in [500]:
        rf = randomforest(max_depth = depth, trees = tree, features = 'sqrt')
        rf.fit(X_train, y_train)
        filename = 'data/models/randomforest_gatein_final.joblib'
        joblib.dump(rf, filename) 

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

metrics_total_test.sort_values(['MAPE', 'MAE']).head(5)

metrics_total_test.to_csv('data/results/randomforest_test_gatein_final.csv')
metrics_total_train.to_csv('data/results/randomforest_train_gatein_final.csv')
metrics_total_test

metric_best
metric_best['model'] = 'gate_in'
metric_best['depth'] = 9
metric_best = metric_best.append(pd.read_csv('data/results/randomforest_best_model_metrics.csv'))
metric_best.to_csv('data/results/randomforest_best_model_metrics.csv')

feature_importance = (pd.DataFrame(X_train.columns, rf.feature_importances_)
                        .reset_index()
                        .sort_values('index', ascending=0))
feature_importance

# save output

output = model_booked.ix[list(test.index.values)]
output['y_hat'] = y_hat
output.to_csv('data/results/route_depart_new.csv')
filename = 'data/models/randomforest_depart_final.joblib'
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
                 epochs = 200, batch_size = 500)

X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm)

y_hat = nn.predict(X_test_norm)
y_hat_new = y_hat.flatten()    
y_test = test.reset_index()['y']

m = print_metrics(y_hat_new, y_test)
m['model'] = 'depart'
m

metric_best = metric_best.append(m)
metric_best

metric_best.to_csv('data/results/neuralnetwork_model_metrics.csv')

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


###########################################################################################

#regression tree

from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(max_depth=5, 
                            min_samples_split = 15,
                            min_samples_leaf =5,
                            random_state = 7474
                            )
clf = clf.fit(X_train,y_train)

y_hat_test = clf.predict(X_test)
y_hat_train = clf.predict(X_train)

print_metrics(y_hat_test, y_test)

# print picture of tree
from sklearn.tree import export_graphviz
import graphviz
features = list(X_train.columns)
dot_data = export_graphviz(clf, out_file=None, feature_names = features,class_names=['0','1'],  
                                  filled=True, rounded=True,  special_characters=True)
graphviz.Source(dot_data)

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('tree_graph',view=True)

##################################################################################################

# linear regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print('fitted linear model...')

coef = pd.DataFrame({'names': X_train.columns, 'coeficients': regr.coef_})
y_hat = regr.predict(X_test)

m = print_metrics(y_hat, y_test)
m['model'] = 'depart'
m

metric_best = metric_best.append(m)
metric_best

metric_best.to_csv('data/results/linear_model_metrics.csv')

print(coef)

#############################################################################################################################
# base case
model = customer_clean[(customer_clean['y_receive'] > 10) & (customer_clean['y_receive'] < 100)]
model = model.rename(columns={'y_receive':'y'})

date = max(model['Container Unload From Vessel-Actual']) - datetime.timedelta(days=365)
train = model[model['Container Unload From Vessel-Actual'] < date].drop('Container Unload From Vessel-Actual', axis=1)
test = model[model['Container Unload From Vessel-Actual'] >= date].drop('Container Unload From Vessel-Actual', axis=1)

test_base = (train.groupby(['Carrier','Original Port Of Loading',
                            'Final Port Of Discharge'])['y']
                  .mean().reset_index().rename(columns={'y':'mean'})
                  .merge(test,
                         on=['Carrier','Original Port Of Loading', 'Final Port Of Discharge'],
                         how='right'))
    
test_base['mean'] = round(test_base['mean'], 2)

test_base = test_base[np.isnan(test_base['y']) == False]
test_base = test_base[np.isnan(test_base['mean']) == False]

m = print_metrics(test_base['mean'], test_base['y'])
m['model'] = 'received'
metric_base.append(m)

metric_base = metric_base.append(m)
metric_base.sort_values(' R2')
metric_base.to_csv('data/results/base_model_metrics.csv')
test_base.columns

################################################################################################################################
