# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:46:17 2019

@author: Florian Krempl
"""

#####################
## create rf model ##
#####################

def randomforest(max_depth, trees, criterion = 'mse', features = 'auto'):
    regr = RandomForestRegressor(n_estimators = trees,
                             criterion = criterion,
                             random_state = 1992,
                             max_depth = max_depth,
                             max_leaf_nodes = None,
                             min_samples_split = 2,
                             min_samples_leaf = 10,
                             max_features = features,
                             #min_impurity_decrease = cp,
                             n_jobs = -1,
                             verbose = 100
                             )
    return regr


#####################
## create boosting tree model
#####################

def boost(depth, trees):
    boost = GradientBoostingRegressor(loss = 'ls', learning_rate = 0.01,
                                  n_estimators = trees, max_depth = depth, 
                                  criterion = 'mae', min_samples_leaf = 10,
                                  random_state= 1992, max_features = 2, 
                                  verbose = 100)
    return boost


############################
### get test performance  ##
############################
    
from sklearn import metrics
def print_metrics(y_hat, y_test):
    output = pd.DataFrame({'MAE': [round(np.mean(abs(y_hat-y_test)), 2)],
                   'MAPE': [round(np.mean(abs(y_hat-y_test)/y_test), 4)],
                   'RMSE': [round(np.sqrt(np.mean((y_test - y_hat)**2)), 2)],
                   ' R2': [round(metrics.r2_score(y_test, y_hat),2)]
                   })
    return output

##########################
##  load customer data  ##
##########################
    

def load_data(customers, folder, names):
    
    # create empty dataframe to append loaded csv data later
    customer_new = pd.DataFrame({'Shipper':[], 'Carrier':[],
                               'Original Port Of Loading':[],
                               'Original Port Of Loading Site':[],
                               'Final Port Of Discharge':[],
                               'Final Port Of Discharge Site':[],
                               'Origin Service':[], 
                               'Book Date':[], 'Expected Receipt Date':[], 'Actual Receipt Date': [], 
                               'Gate In Origin-Actual':[], 'Consolidation Date':[],
                               'ETD':[], 'ATD':[],'ETA':[], 'ATA':[],
                               'Container Unload From Vessel-Actual':[],
                               'Equipment Number':[]})
    
    for customer_name in customers:
        # load csv files
        print("start to load", customer_name, "...") 
        customer = pd.read_csv(os.getcwd() + folder + customer_name + '.csv',
                               encoding= 'latin1')
        print("loaded", customer_name, "starting to clean the dataframe")
        # test for unnecessary columns
        if customer.columns[0] == 'Unnamed: 0':
            customer = customer.drop(columns='Unnamed: 0')
            print('droped unnecessary index column')
        # get right column names    
        customer.columns = column_names
        # get real carrier
        customer["Carrier"] = np.where(customer['VOCC Carrier'].isna,
                                       customer["CBL Number"],
                                       customer["VOCC Carrier"])
        # get rid of missing data - select right columns
        customer = (customer.loc[customer['Final Port Of Discharge Site'] == 'UNITED STATES']
                            .loc[customer['Container Unload From Vessel-Actual'].notna()]
                            .loc[:,['Shipper', 'Carrier',
                                    'Original Port Of Loading',
                                    'Original Port Of Loading Site',
                                    'Final Port Of Discharge',
                                    'Final Port Of Discharge Site',
                                    'Origin Service', 'Book Date',
                                    'Expected Receipt Date', 'Actual Receipt Date',
                                    'Latest Receipt Date',
                                    'Gate In Origin-Actual', 'Consolidation Date',
                                    'ETD', 'ATD','ETA', 'ATA',
                                    'Container Unload From Vessel-Actual',
                                    'Equipment Number']])
        # attach customer name column
        customer['customer'] = customer_name
        # customer['Final Port Of Discharge'] = customer['Final Port Of Discharge'].str.title()
        # append to customer_clean
        print("appending", customer_name, "to customer_clean", "\n") 
        customer_new = customer_new.append(customer)
        # get new index - let's start clean from here
    customer_new = customer_new.reset_index(drop=True)
        
    return customer_new

###########################
## prepare predict data ##
###########################

# model booking date:
    
def get_features_booking(df):
    df['quarter'] = df['Expected Receipt Date'].dt.quarter
    df['ETP'] = (df['Expected Receipt Date'] - df['Book Date']).dt.days   # expected time between booking and receival
    df['Origin Service'] = np.where(df['Origin Service'] == 'CFS', 1, 0)
    
    # load other metrics
    new_transports = (df.merge(stat_pod, on=['Final Port Of Discharge', 'customer'])
                        .merge(stat_route, on=['Carrier', 'Original Port Of Loading', 'Final Port Of Discharge'])
                        .merge(stat_quarter, on=['quarter'])
                        .merge(stat_poo, on=['Original Port Of Loading', 'Shipper']))
    
    # select features
    features = ['pod_mean', 'pod_std', 'poo_mean', 'poo_std', 'route_mean',
                'Origin Service', 'holiday', 'quarter_z', 'ETP']
    new_transports = new_transports[features]
    return new_transports






