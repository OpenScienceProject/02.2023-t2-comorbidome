#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:07:36 2023

@author: Damien Brisou
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


# PARAMETERS___________________________________________________________________
proj_path = '/home/image_in/dev_ops/02.2023-t2-comorbidome'

# Plot parameters:
figsize = (10, 7)

# Load metadatas:_________________________________________________________
try:
    proj_path = Path(sys.argv[1])
except:
    proj_path = Path(proj_path)
finally:
    input_path = proj_path.joinpath('data', 'outputs', 'python')

with open(input_path.joinpath('_utils', 'variables.json')) as json_f:
    var_class = json.load(json_f)
with open(input_path.joinpath('_utils', 'parameters.json')) as json_f:
    params = json.load(json_f)
    
int_col = ['Age', 'Polymedication', 'duration_days']
var_class['COMORBIDITES'].remove('Polymedication')
binary_col = [*var_class['COMORBIDITES'], 
              *var_class['SCLERODERMIE'], 'Female', 'Male']
cat_col, boolean_col = ['Sex'], ['decede']
polymedication_threshold = params['polymedication_threshold']
#_________________________________________________________________________

# Plot names:
sgp_dct0 = { "Patient's field":
            {'Global': 'Global', 'Tabagism': 'Tabagism', 'Osteoporosis':'Osteoporosis', 
             'Obesity':'Obesity', 'Female': 'Female', 'Male': 'Male',} 
            }   
    
sgp_dct1 = {'Cardiovascular diseases':
            {'Global': 'Global', 'Stroke': 'Stroke', 'Myocardial Infarction': 'MI', 
           'Lower limbs Arterial Disease': 'LEAD','Cardiopathy': 'Cardiopathy', } 
            }   
    
sgp_dct2 = {'Chronic diseases':
                {'Global': 'Global', 'COPD': 'COPD', 'Hemiplegia': 'Hemiplegia',
                 'Diabetes': 'Diabetes' , 'HIV': 'HIV', 
                 'CKD': 'CKD', 'Dementia':'Dementia', 'Neoplasia': 'Neoplasia' }
                }  
    
sgp_dct3 = {'Sclerodermic involvement': 
            {'Global': 'Global', 'Cardiac': 'Cardiovascular_involvement', 
                'Pulmonary': 'Lung_involvement','Digestive': 'Digestive_involvement',
                'Renal': 'Renal_involvement'}
            }
 
sgp_dct4 = { 'Other' :  
                {'Global': 'Global', 'Hepatic': 'Liver disease',  
        'Gastric ulcer': 'Gastric ulcer', 'Anxiety': 'Anxiety', 
        f'Polymedication>{polymedication_threshold}':f'Polymedication>{polymedication_threshold}',}}
    
sgps = (sgp_dct0 , sgp_dct1, sgp_dct2, sgp_dct3, sgp_dct4)

#______________________________________________________________________

## Format data
def load_format_data(path):
    df_ = pd.read_csv(path, sep=';')
    # Set variables types:
    for var_type, el_col in {'category': cat_col, 'int8': binary_col, int: int_col, 
                              bool: boolean_col}.items():
        for col_ in el_col:
            df_[col_] = df_[col_].astype(var_type)
    return df_, df_.columns

df, col = load_format_data(input_path.joinpath('formatted_data.csv'))                

## Explore variables:
# Count number and percentages of positive population:
from sklearn.preprocessing import LabelEncoder
stats_binary_df= pd.DataFrame()
for col_ in [*binary_col, *boolean_col, *cat_col]:
    df_ = df.copy()
    encoder = LabelEncoder()
    df_[cat_col[0]] = encoder.fit_transform(df_[cat_col[0]])
    nb_ = pd.Series(dtype='float64')
    perc_ = pd.Series(dtype='float64')
    nb_d = pd.Series(dtype='float64')
    perc_d = pd.Series(dtype='float64')
    
    nb_[col_] = len(df_[df_[col_]==1])
    perc_[col_] = df_[col_].sum()/len(df_[col_])*100
    
    nb_d[col_] = len(df_[(df_[col_]==1)&(df_['decede']==1)])
    perc_d[col_] = nb_d[col_]/nb_[col_]*100
    
    pre_df_ = pd.concat([nb_, perc_, nb_d, perc_d], axis=1)
    stats_binary_df= pd.concat([stats_binary_df, pre_df_])
    
stats_binary_df.columns = ['Patients_Nb', 'prevalence_%', 'deceased_Nb', 'deceased_%']

# Stats on numerical variables:
stats_int_df = df[[*int_col]].describe()

## Plot Data:
from sksurv.nonparametric import kaplan_meier_estimator

# Compute and plot Kaplan Meier function with Kaplan-Meier estimator:
def km_groups(df_, sgp_dct):
    group_name, dct = list(sgp_dct)[0], list(sgp_dct.values())[0]
    fig, ax = plt.subplots()
    
    for sgp in dct:
        if sgp == 'Global':
            sub_df = df_
        else:
            sub_df = df_[df_[dct[sgp]] == 1]
            
        time_days, survival_prob = \
            kaplan_meier_estimator(sub_df['decede'], sub_df['duration_days'])
        time_years = time_days / 365
        plt.plot(time_years, survival_prob, label=sgp)

    plt.title(label=f'Survival curve function estimated by Kaplan-Meier in {group_name}')
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time (years)")
    plt.yticks(ticks=np.arange(0, 1, step=0.2))
    plt.legend(loc='best')
    fig.set_size_inches(figsize)
    
    graph_path = input_path.joinpath('graphs')
    graph_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(graph_path.joinpath(f'{group_name}'))

    plt.close()

for sgp_dct in sgps:
    km_groups(df, sgp_dct)

## Check correlations between variales:

# Save stats:
table_path = input_path.joinpath('tables')
table_path.mkdir(parents=True, exist_ok=True)
stats_binary_df['prevalence_%'].round(2).to_csv(input_path.
                                joinpath('_utils', 'comorbidome_plot.csv'), sep=';')
stats_binary_df = pd.concat([stats_binary_df, stats_binary_df.describe()])
stats_binary_df['Patients_Nb'] = stats_binary_df['Patients_Nb'].astype(int)
stats_binary_df.round(2).to_csv(table_path.joinpath('Var_stats(binary).csv'), sep=';')
stats_int_df.round(2).to_csv(table_path.joinpath('Var_stats(numeric).csv'), sep=';')
