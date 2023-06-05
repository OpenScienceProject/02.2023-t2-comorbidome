# Steps to reproduce:

## Project structure
    .
    ├── data               
    │   ├── inputs         
    │   └── outputs        
    └── python
        └── ... (scripts.py)         

## A- Data cleaning and formatting:
- After a manual cleaning in excel ('recueil VRAI.xlsx' > 'clean_data.csv')
- Open the script python/1-prepare_data.py with an IDE and modify parameters or launch it on command line with the root path of the project:
'''
python3 python/1-prepare_data.py '/path/to/project'
'''
It will create a file 'formatted_data.csv' in outputs/python, and .json files that will store parameters

Parameters than can be adapted in the script:
'''
replace_dct # python dictionnary taht will replace column names
df_modif_dct # modify columns columns: 'merge': merge several columns in one; 'create': create new binary columns; 'correct_inputs': corrects rows values
df_col_ops # modify row values over columns: 'data_type': modify data type of the column; 'sup': set True if superiror to ..; 'equal': set True if equal to ..
'''

## B- Statistical analysis:
- Open the script python/2-Statistical_analysis_plots.py with an IDE and modify parameters or launch it on command line with the root path of the project:
'''
python3 python/2-Statistical_analysis_plots.py '/path/to/project'
'''

Parameters than can be adapted in the script:
'''
sgp_dct0
... 
sgp_dctn
: python dictionnaries where you can define plot groups for survival plots, the first key is the name of the group, then in the sub-dictionnary, the first name refers to a column of the dataframe and the second is the title in the plot.

'''
The script will result in the creation of 2 tables: data/ouputs/python/tables/Var_stats(binary).csv and .../Var_stats(numeric).csv, containing the statistics of binary or numerical variables;
and a table in _utils/comorbidome_plot.csv containing the prevalence for each variable.
It will also plot simple survival curves of each group difined in parmaters

## C- Creation of a survival regression model and a comorbidome plot
- Open the python/script 3-Survival_model_tuning.py with an IDE and modify parameters or launch it on command line with the root path of the project:
'''
python3 python/3-Survival_model_tuning.py '/path/to/project'
'''
Parameters than can be adapted in the script:
'''

'''

It will perform a regression model using

## D- Hyperparameters tuning



# Data save location:
Immunoconcept server > Partage_Patrick_Blanco > Damien Brisou > Colabs > 02.2023- Comorbidome > data

# Code save location (public):
https://github.com/OpenScienceProject/02.2023-t2-comorbidome
