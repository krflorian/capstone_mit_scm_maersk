#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:20:41 2018

@author: drx
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

os.chdir('/media/shareddata/MIT/Capstone')
os.getcwd()

# read port data
imports = pd.read_excel('data/port data/us_imports_1.xlsx')

for i in range(2,50): 
    imports = imports.append(pd.read_excel('data/port data/us_imports_' +str(i)+'.xlsx'), ignore_index = True)
    print('appended us_imports_' + str(i))

imports.to_csv("data/port data/us_imports_total.csv")

# get date to right format

imports['Arrival Date'] = pd.to_datetime(imports['Arrival Date'])

# get statistics
port_summary = (imports.groupby(['Port of Unlading', 'Arrival Date'])['Number of Containers']
                       .agg(np.sum)
                       .reset_index())

port_summary['year'] = port_summary['Arrival Date'].dt.year
port_summary['week'] = port_summary['Arrival Date'].dt.week

port_max_cap = (port_summary.groupby(['Port of Unlading', 'year'])['Number of Containers']
                            .agg(np.max)
                            .reset_index())

port_max_cap.columns = ['Port of Unlading', 'year', 'max_cap']

port_summary = port_summary.merge(port_max_cap,
                                  on = ['Port of Unlading','year'],
                                  how = 'inner')

port_summary['cap'] = round(port_summary['Number of Containers'] / port_summary['max_cap'], 2)

#get port city + name + state

port_summary = (port_summary.join(port_summary['Port of Unlading'].str.split(',', expand=True))
                            .drop('Port of Unlading', axis=1))

port_summary.columns = ['Arrival Date', 'Number of Containers', 'year', 'week',
                        'max_cap', 'cap', 'Port Name', 'City', 'State']

port_summary['City'] = port_summary['City'].str.strip()

# get coast

west_coast = ['Los Angeles', 'Long Beach', 'San Pedro',
              'Oakland', 'Seattle', 'Tacoma']
east_coast = ['Savannah', 'Newark', 'Houston', 'New York', 
              'Norfolk']

port_summary['coast'] = np.where(np.isin(port_summary['City'], west_coast),
            'west coast', 'east coast')

# save statistics
port_summary.to_csv("data/port_summary_us.csv")





####################
# plot capacity   ##
####################

plot = port_summary[port_summary['Port of Unlading'] == 'The Port of Los Angeles, Los Angeles, California']

# plot histogram of number per day
plt.hist(plot['cap'])
plt.show()

# plot timeseries per day
plt.plot(plot['Arrival Date'], plot['cap'])
plt.show()

# plot timeseries per months
plot['month'] = plot['Arrival Date'].dt.month
plot['year'] = plot['Arrival Date'].dt.year

plot2 = plot.groupby(['year', 'month'])['cap'].agg(np.mean)
plot2 = plot2.reset_index()
plot2['date'] = pd.to_datetime(plot2['year'].map(str) + '/' + plot2['month'].map(str) + '/01', format = '%Y/%m/%d')

plt.plot(plot2[plot2['year'] == 2015]['month'], plot2[plot2['year'] == 2015]['cap'])
plt.plot(plot2[plot2['year'] == 2016]['month'], plot2[plot2['year'] == 2016]['cap'])
plt.plot(plot2[plot2['year'] == 2017]['month'], plot2[plot2['year'] == 2017]['cap'])
plt.plot(plot2[plot2['year'] == 2018]['month'], plot2[plot2['year'] == 2018]['cap'])
plt.show()





# load us port capacity statistics
port_cap = pd.read_csv('data/statistics/summary_ports_us.csv',
                       sep = ',', encoding='latin1')
port_cap['Arrival Date'] = pd.to_datetime(port_cap['Arrival Date'], format = '%d.%m.%Y')

#port_cap = port_cap[['Arrival Date', 'Number of Containers', 'cap', 'Port Name', 'City', 'State', 'coast']]

georgia = port_cap[port_cap['Port Name'] == 'Georgia Ports Authority']
georgia.index = georgia['Arrival Date']
plt.plot(georgia['cap'])

georgia['Arrival Date'].dt.dayofyear

georgia['rolling'] = georgia.rolling(window = 3, center=False)['cap'].mean()
georgia



plt.plot(georgia_series.rolling(3, center = True).mean())
port_cap.index = port_cap['Arrival Date'].dt.dayofyear
port_forecast = port_cap.groupby(['Port Name', 'year'])['cap'].rolling(3, center = True).mean()
port_forecast = port_forecast.reset_index()
port_forecast.columns = ['Port Name', 'year', 'doy', 'cap']
port_forecast = port_forecast.groupby(['Port Name', 'doy'])['cap'].mean()
port_forecast = port_forecast.reset_index()
port_forecast


plt.plot(port_forecast[port_forecast['Port Name'] == 'Georgia Ports Authority']['cap'])

