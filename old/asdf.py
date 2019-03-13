# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:58:32 2019

@author: kr_fl
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib as plt
%matplotlib auto

os.chdir('E:\MIT\Capstone')

results = pd.read_csv("data/results/book_test_new.csv", index_col = 'Unnamed: 0')

results['residuals'] = results['y_hat']-results['y_book']

sns.jointplot(x='y_book',y='residuals',kind='hex', data=results)

sns.distplot(model_received_ready['y_receive'])

plot_data = customer_clean[np.isnan(customer_clean['poo_rec']) == False]
sns.distplot(plot_data[plot_data['Origin Service'] == 1]['poo_rec'])
sns.distplot(plot_data[plot_data['Origin Service'] == 0]['poo_rec'])
plt.show()


customer_clean['poo_rec'] = (customer_clean['ATD']-customer_clean['Actual Receipt Date']).dt.days






