#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:07:36 2023

@author: Damien Brisou
"""

from pathlib import Path
import pandas as pd
import sys
import numpy as np
import json

# Parameters:_______________________________________________________
proj_path = '/home/image_in/dev_ops/02.2023-t2-comorbidome'
csv_file = 'clean_data.csv'

set_lost_patients_alive = True
polymedication_threshold = 4

replace_dct = {'Debut': 'Start', 'Fin':'End', 
               'polymédication ':'Polymedication', 
               'Sexe': 'Sex', 'IDM': 'MI', 'cardiopathie': 'Cardiopathy', 
               'AOMI': 'LEAD', 'HTA': 'HTN', 'AVC/AIT': 'Stroke',
                'Démence': 'Dementia', 'tabac': 'Tabagism', 'BPCO': 'COPD', 
                'Ulcère gastrique': 'Gastric ulcer',
                'Maladie hépatique/OH': 'Liver disease', 'Diabète sucré': 'Diabetes', 
                'Hémiplégie': 'Hemiplegia', 'IRC': 'CKD',
                 'VIH': 'HIV', 'Anxiété': 'Anxiety',
                'Ostéoporose': 'Osteoporosis', 'Obésité': 'Obesity',
               'Cutanéo-articulaire': 'Skin_involvement',
               'Cardio-circulatoire': 'Cardiovascular_involvement', 
               'poumon': 'Lung_involvement', 
               'rein ': 'Renal_involvement', 'digestive': 'Digestive_involvement'}

df_modif_dct = {
    'merge': {
        'COMORBIDITES': (
            ['Tumeur solide', 'Leucémie', 'Lymphome'], 'Neoplasia'
                        )},
    'create': {
        'COMORBIDITES': (f'Polymedication>{polymedication_threshold}',),
        'TERRAIN': ('Female', 'Male') },
    'correct_inputs': { 
        'Sex': [('femme ', 'femme'), ('femmea', 'femme'), ('homme ', 'homme')]
                }
            }

df_col_ops = {
    'data_type': {int: ['Polymedication']},
    'sup': [('Polymedication', f'Polymedication>{polymedication_threshold}',
             polymedication_threshold)],
    'equal': [('Sex', 'Female', 'femme', ),
              ('Sex', 'Male', 'homme', )]
    }

#__________________________________________________________________
try :
    proj_path_ = Path(sys.argv[1])
except:
    proj_path_ = proj_path
finally:
    print(f'Project path: {proj_path_}')
    input_path = Path(f'{proj_path_}/data/inputs')
    output_path = Path(f'{proj_path_}/data/outputs/python')

## Save parameters:
params = dict({'set_lost_patients_alive': set_lost_patients_alive, 
               'polymedication_threshold': polymedication_threshold})
with open(output_path.joinpath('_utils', 'parameters.json'), 'w') as f:
    f.write(json.dumps(params))

## Format data
path = input_path.joinpath(csv_file)

def load_format_df(path, replace_dct, df_modif_dct, df_col_ops):
    df_ = pd.read_csv(path, sep=',')
    var_categories = df_.iloc[0].unique()
    
    # Change column names according to replace_dct:
    col = df_.columns
    for k, v in replace_dct.items():
        col = col.str.replace(k, v)
    df_.columns = col
        
    # Modify dataframe according to df__modif_dct:
    for op, sub_dct in df_modif_dct.items():
        if op == 'merge':
            for category, data in sub_dct.items():
                old_cols, new_col = data
                df_[new_col] = df_.loc[1:,old_cols].sum(axis=1)
                df_ = df_.drop(columns=old_cols)
                df_[new_col][df_[new_col] != 0] = '1'
                df_[new_col][df_[new_col] == 0] = '0'
                df_[new_col].iloc[0] = category
        # Create a new columns with df__modif_dct data:
        if op == 'create':
            for category, str_datas in sub_dct.items():
                for str_data in str_datas:
                    df_[str_data] = '0'
                    df_[str_data].iloc[0] = category
        if op == 'correct_inputs':
            for col, corrections in sub_dct.items():
                for corr in corrections:
                    old_str, new_str = corr
                    df_[col].replace(old_str, new_str, inplace=True)
    
    # Defines subcateories of variales:
    var_class = dict()
    for category in var_categories:
        var_class[category] = list(df_.loc[:,df_.iloc[0] == category].columns)
    with open(output_path.joinpath('_utils', 'variables.json'), 'w') as f:
        f.write(json.dumps(var_class))
    
    df_=df_.drop([0]) # Remove first line which belongs to column infos
    
    # Operations on column values:
    for op, sub_ls in df_col_ops.items():
        if op == 'data_type':
            for dtype, cols in sub_ls.items():
                df_[cols] = df_[cols].astype(dtype)
        if op == 'sup':
            for el in sub_ls:
                source_col, target_col, value = el
                df_.loc[df_[source_col]>value, target_col] = '1'
        if op == 'equal':
            for el in sub_ls:
                source_col, target_col, value = el
                df_.loc[df_[source_col]==value, target_col] = '1'
    
    # Deal witn NaN values:
    print(f'Presence of Nan values: {df_.isnull().values.any()}')
    df_.fillna('0', inplace=True)
    print(f'Presence of Nan values after fill: {df_.isnull().values.any()}')
    return df_

df = load_format_df(path, replace_dct, df_modif_dct, df_col_ops)
    
## Format time columns to date/ time, and create a duration column:
from lifelines.utils import datetimes_to_durations

# Set all patient not dead as right censured (if set_lost_patients_alive is True):
if set_lost_patients_alive == True:
    df.loc[df.decede == '0', 'End'] = np.nan

# Transform data in date and time and compute duration:
df.End = df.End.replace('0', np.nan)
df[['Start', 'End']] = df[['Start','End']].apply(pd.to_datetime, dayfirst=True)
# Get all dates:
start_date = min(pd.to_datetime(df.Start.unique()))
end_date = max(pd.to_datetime(df.End.unique()[1:]))
print(f'Follow up from {start_date} to {end_date}')
# Define duration outcome ase on dates:
df['duration_days'], df['EndDate'] = datetimes_to_durations(
    df.Start, df.End, freq='D', fill_date=end_date)

# Save:
df.to_csv(output_path.joinpath('formatted_data.csv'), sep=';', index=False)