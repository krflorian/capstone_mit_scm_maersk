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
from sklearn.externals import joblib

os.chdir('E:\MIT\Capstone')
os.getcwd()

# check if stats are loaded in 
if 'stat' not in locals():
    exec(open('scripts/functions.py').read())
    exec(open('scripts/setup_stats.py').read())

##########################################################################################
# get relevant variables
model_not_ready = True
while model_not_ready:
    model = input('please input the model name you want to train \n  booked \n received \n gate_in \n depart \n')
    warmstart = input('warm start? \n yes \n no \n')
    print('start to load model data...')
    if (model == 'departed'):
    # get model departed
        model_departed, features = get_features_departed(customer_clean, model_training=True)
        model_ready = model_departed.dropna(subset = features)
        model_not_ready = False
        print('model departed ready...')
        # get model received
    elif (model == 'received'):
        model_received, features = get_features_received(customer_clean, model_training=1)
        model_ready = model_received.dropna(subset = features)
        model_not_ready = False
        print('model received ready...')
    # get model gate_in
    elif (model == 'gate_in'):
        model_gate, features = get_features_gate(customer_clean, model_training=1)
        model_ready = model_gate.dropna(subset = features)
        model_not_ready = False
        print('model gate_in ready...')
    # get variables booked
    elif (model == 'booked'):
        model_booked, features = get_features_booking(customer_clean, model_training=1)
        model_ready = model_booked.dropna(subset = features)
        model_not_ready = False
        print('model booked ready...')
    # check if model is loaded
    else: 
        print('please input correct model name')
# select relevant columns
model_ready = model_ready[features]
model_ready = model_ready.drop('Container Unload From Vessel-Actual', axis = 1)

##########################################################################################
##########################################################################################

# select data for model
model_ready = model_ready.rename(columns={'y_depart':'y', 'y_book':'y', 'y_receive':'y', 'y_gate': 'y'})

X = model_ready.drop('y', axis=1)
y = model_ready['y']
tree = 500

# select parameters for model
if (model == 'booked'):
    depth = 10
elif (model == 'received'):
    depth = 7   
elif (model == 'gate_in'):
    depth = 9 
elif (model == 'depart'):
    depth = 5 

# train random forest model
print('begin training model...')
# load old model
filename = 'data/models/randomforest_' + model +'_new.joblib'

if warmstart == 'yes':
    rf = joblib.load(filename)
    rf.set_params(warm_start=True)
else:
    rf = randomforest(max_depth = depth, trees = tree, features = 'sqrt')

# train model
rf.fit(X, y)

# save model
joblib.dump(rf, filename)

