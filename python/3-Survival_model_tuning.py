#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:23:41 2023

@author: image_in
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import cross_val_score


# PARAMS______________________________________________________________________
input_path = Path(
    '/home/image_in/dev_ops/02.2023-t2-comorbidome/data/outputs/python')

# Load metadatas:
with open(input_path.joinpath('_utils', 'variables.json')) as json_f:
    var_class = json.load(json_f)
with open(input_path.joinpath('_utils', 'parameters.json')) as json_f:
    params = json.load(json_f)

# Model features: "breslow" (semi_parametric), "spline" (parametric), or "piecewise"
parametric_model = False
n_baseline_knots = 5  # for parametric model
# Evaluation parameters:
cv_split = 6  # Nb of splits for cross validation

# Define extrem values of prevalence % and number to exclude from analysis:
prev_max, prev_min = 95, 2

# Plot parameters:
figsize = (17, 16)
figsize_surv = (10, 7)
figsize_comor = (190, 190)
comor_max_plot = 2

# Variables management:_____________________________________________________
int_col = ['Age', 'Polymedication', 'duration_days']
var_class['COMORBIDITES'].remove('Polymedication')
binary_col = [*var_class['COMORBIDITES'], *var_class['SCLERODERMIE'],
              'Female', 'Male']
cat_col, boolean_col = ['Sex'], ['decede']
polymedication_threshold = params['polymedication_threshold']

# Variable matrice X:
drop_col = ['Female', 'Male', 'Start', 'End', 'EndDate',
            'decede', 'duration_days']
# Outcomes:
Y_col = ['decede', 'duration_days']

# Define groups of diseases:
groups_dct = {
    'Sclerodermic involvement': ['Cardiovascular_involvement',
                                 'Lung_involvement', 'Renal_involvement',
                                 'Digestive_involvement'],
    'Cardio-vacular': ['Stroke', 'MI', 'LEAD', 'Cardiopathy', 'HTN'],
    'Chronic': ['COPD', 'Hémiplegia', 'Liver disease',
                'Diabetes', 'HIV', 'CKD', 'Dementia', 'Neoplasia'],
    'Field': ['Tabagism', 'Osteoporosis', 'Obesity', 'Sex'],
    'Other': ['Gastric ulcer', 'Hemiplegia', 'Anxiety', 'Polymedication',
              f'Polymedication>{polymedication_threshold}']
}

colors_ls = ['royalblue', 'green', 'purple',
             'turquoise', 'darkorange', 'magenta', 'yellow']

# PARAMS______________________________________________________________________

# Save Parameters:
params['parametric_model'] = parametric_model
params['n_baseline_knots'] = n_baseline_knots
params['cv_split'] = cv_split
params['prev_max'] = prev_max
params['prev_min'] = prev_min

with open(input_path.joinpath('_utils', 'parameters.json'), 'w') as f:
    f.write(json.dumps(params))

# Load and format data
# Deals with variale selection:


def exclude_table():  # Identifies extrem values:
    df_ = pd.read_csv(input_path.joinpath('tables', 'Var_stats(binary).csv'),
                      index_col=0, sep=';')
    df_['extreme%'] = False
    df_.loc[(df_['prevalence_%'] > prev_max) |
            (df_['prevalence_%'] < prev_min), 'extreme%'] = True
    df_.to_csv(input_path.joinpath(
        'tables', 'Excluded_variables.csv'), sep=';')
    return df_


def load_format_data():
    df_ = pd.read_csv(input_path.joinpath('formatted_data.csv'), sep=';')
    df_comorbidome_plot = pd.read_csv(input_path.
                                      joinpath(
                                          '_utils', 'comorbidome_plot.csv'),
                                      sep=';', index_col=0).drop(labels='decede')
    df_comorbidome_plot['groups'] = 'None'
    for g, ls in groups_dct.items():
        for el in ls:
            df_comorbidome_plot.loc[el, 'groups'] = g
    for col in drop_col:
        if col in df_comorbidome_plot.index:
            df_comorbidome_plot = df_comorbidome_plot.drop(labels=col)

    # Set variables types:
    for var_type, el_col in {'category': cat_col, 'int8': binary_col, int: int_col,
                             bool: boolean_col}.items():
        for col_ in el_col:
            df_[col_] = df_[col_].astype(var_type)
    # Convert categorical in binary variable:
    encoder = LabelEncoder()
    df_[cat_col[0]] = encoder.fit_transform(df_[cat_col[0]])
    exclude_df = exclude_table()
    # Exclude extrem prevalences:
    exclude_ls = list(exclude_df[exclude_df['extreme%']].index)
    for el in exclude_ls:
        # from df:
        if el in df_.columns:
            df_ = df_.drop(columns=el)
        # from df_comrbidome:
        if el in df_comorbidome_plot.index:
            df_comorbidome_plot = df_comorbidome_plot.drop(index=el)
    return df_, df_comorbidome_plot


df, df_comorbidome_plot = load_format_data()

# Plot correlation matrix:


def corr_matrix(X):
    corr = X.corr(method='pearson')
    plt.figure(figsize=figsize)
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)

    plt.savefig(input_path.joinpath('graphs', 'corr_matrix.png'))


# Define matrices:
# Variables:
X = df.drop(columns=drop_col)
corr_matrix(X)  # Plot correlations
# Outcomes:
Y = df[Y_col]
# Clean dataframe:
df_hr = pd.concat([X, Y], axis=1)

# Cox Proportional-Hazards Model:
# Validation metrics funcion:


def cross_val_scores(df_):
    df_copy = df_.copy()
    X = df_copy.drop('duration_days', axis=1)
    Y = df_copy.pop('duration_days')  # keep only duration
    if parametric_model:
        coxph_adapt = sklearn_adapter(WeibullAFTFitter, event_col=Y_col[0])
    else:
        coxph_adapt = sklearn_adapter(CoxPHFitter, event_col=Y_col[0])

    wf = coxph_adapt()
    scores = cross_val_score(wf, X, Y, cv=cv_split)
    return scores

# Cox model with lifelines:


def cox_hr_life(df_hr):
    if parametric_model:
        print("Parametric model")
        coxph = WeibullAFTFitter()
    else:
        print("Semi-parametric model")
        coxph = CoxPHFitter()
        # will output violations of the proportional hazard assumption

    coxph.fit(df_hr, event_col=Y_col[0], duration_col=Y_col[1])
    # summary table of resuls:
    summary = coxph.summary
    # Model metrics for fine tuning:
    with open(input_path.joinpath('_utils', 'parameters.json')) as json_f:
        params = json.load(json_f)
    metrics = pd.DataFrame({'parametric': params['parametric_model'],
                            'param_alive': params['set_lost_patients_alive'],
                            'param_prev_max': float(params['prev_max']),
                            'param_prev_min': float(params['prev_min'])}, index=[0])
    metrics['concordance'] = coxph.concordance_index_
    metrics['likelihood ratio'] = str(
        coxph.log_likelihood_ratio_test().summary)
    cross_vals = cross_val_scores(df_hr)
    metrics['cross_validation_scores'] = np.nanmean(cross_vals)
    if parametric_model:
        df_comorbidome_plot['log(hr)'] = summary['exp(coef)']
        df_comorbidome_plot['p'] = summary['p']
        df_comorbidome_plot['p_sign'] = df_comorbidome_plot['p'] < 0.05
    else:
        metrics['asumptions'] = len(coxph.check_assumptions(df_hr))
        df_comorbidome_plot['log(hr)'] = summary['exp(coef)']
        df_comorbidome_plot['p'] = summary['p']
        df_comorbidome_plot['p_sign'] = df_comorbidome_plot['p'] < 0.05
        # will output violations of the proportional hazard assumption

    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    plt.title(label='Hazard ratios estimated by Cox Proportional-Hazards Model')
    coxph.plot()

    # Save:
    plt.savefig(input_path.joinpath('graphs').
                joinpath(f'Cox_multivariate_regression_model(parametric={parametric_model})'))
    summary.round(4).to_csv(input_path.joinpath(
        'tables',
        f'Cox_multivariate_regression_model(parametric={parametric_model}).csv'),
        sep=';')
    save_metrics = input_path.joinpath(
        'tables').joinpath('Cox_model_metrics.csv')
    if save_metrics.exists():
        metrics.round(3).to_csv(save_metrics, sep=';',
                                mode='a', index=False, header=False)
    else:
        metrics.round(3).to_csv(save_metrics, sep=';', index=False)
    plt.close()

    return coxph


coxph = cox_hr_life(df_hr)


# Make predictions with the model:
def predict_plot_surv_func(X):
    pred_surv = coxph.predict_survival_function(X)
    pred_mort = coxph.predict_cumulative_hazard(X)

    for title, df in zip(['survival', 'cumHR'], [pred_surv, pred_mort]):
        fig, ax = plt.subplots(figsize=figsize_surv)
        ax.set_title(f"Prediction of patient's {title}")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"{title}")
        for col in df.columns:
            df['time_years'] = df.index/365
            ax.plot(df['time_years'], df[col], label=col)
        plt.legend()
        plt.savefig(input_path.joinpath('predictions', f'{title}_plot'))
        plt.close()

    if parametric_model:
        expec = coxph.predict_expectation(X)
        summary = pd.concat([expec], axis=1)
        summary.columns = ['lifetime expect (days)']
    else:
        expec = coxph.predict_expectation(X)
        hr = coxph.predict_partial_hazard(X)
        summary = pd.concat([expec, hr], axis=1)
        summary.columns = ['lifetime expect (days)',
                           'hazard ratio']
    summary['True lifetime (days)'] = Y['duration_days']
    summary = pd.concat([summary, X], axis=1)
    summary.loc[summary.Sex == 1, 'Sex'] = 'M'
    summary.loc[summary.Sex == 0, 'Sex'] = 'F'
    summary.round(3).to_csv(input_path.joinpath(
        'predictions', 'Model_predictions.csv'), sep=';')


predict_plot_surv_func(X.sample(n=15))

# Comorbidome plot:


def comoridome_polar_plot(df_):
    prev = df_['prevalence_%']*7000
    hr = 1/(df_['log(hr)'])
    hr = hr.clip(upper=comor_max_plot)
    hr_p = hr[df_['p_sign']]
    groups = df_['groups']

    fig, ax = plt.subplots(figsize=figsize_comor, subplot_kw=dict(polar=True))

    # Parameters
    ax.set_rmax(max(hr))
    ax.set_rticks([])  # Less radial ticks
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    colors = dict(zip(
        groups.unique(), colors_ls[0:len(groups.unique())])
    )

    # Plot:
    # Points:
    angle = pd.DataFrame()
    angle_disp = 2 * np.pi * np.arange(0, 1, 1/len(hr))
    i = 0
    for g in groups.unique():
        idxs = df_.index[df_['groups'] == g]
        hr_ = hr.loc[idxs]
        shape = hr_.shape[0]
        angle_ = angle_disp[i:(i+shape)]
        scatter = ax.scatter(angle_, hr_, c=colors[g], alpha=0.5,
                             label=g, s=prev[idxs])
        angle_ = pd.Series(angle_, index=(idxs))
        angle = pd.concat([angle, angle_], axis=0)
        i += shape

    # Plot the center:
    ax.plot(0, 0, marker='x', markersize=150,
            markeredgecolor='black', markeredgewidth=10)

    # Add size legend of prevalence:
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, num=9)
    # starts at 20000, each step increases of 20000
    handles = [handles[0], handles[2], handles[4]]
    labels = ['5 %', ' 20 %', '   35 %']
    legend1 = plt.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.48, 0.01, 0.5, 0.5),
                         title="Prevalence", fontsize=100,
                         title_fontsize=150, labelspacing=4, facecolor='lightgray', borderpad=3)
    ax.add_artist(legend1)

    # Add legends for Hazard ratio:
    import matplotlib.lines as mlines
    cross = mlines.Line2D([], [], color='grey', marker='x', markersize=120,
                          markeredgecolor='black', label='Risk of mortality (1/log(HR) = 0)',
                          markeredgewidth=10)
    dot_line = mlines.Line2D([], [], color='black', label='       1/log(HR) = 1',
                             linestyle='dotted', linewidth=20)
    legend2 = plt.legend(handles=[cross, dot_line],
                         title="Mortality correlation (HR)", fontsize=120,
                         loc='lower left', bbox_to_anchor=(0.1, 0.05, 0.5, 0.5),
                         title_fontsize=150, labelspacing=4, facecolor='lightgray', borderpad=2)
    ax.add_artist(legend2)

    # Add statistical significance:
    angle_p = angle.loc[hr_p.index]
    ax.scatter(angle_p, hr_p, marker="*", s=30000, c='red', alpha=0.75)
    marker = mlines.Line2D([], [], color='red', marker='*', markersize=150,
                           label='p < 0.05')
    ax.legend(handles=[marker], fontsize=100,
              loc='lower left', bbox_to_anchor=(0.1, 0.01, 0.5, 0.5),
              title="Significativity", title_fontsize=150, facecolor='lightgray')

    # Image text annotations:
    for i, (label, _) in enumerate(hr.items()):
        x = angle.loc[label]
        y = hr.loc[label]
        ax.annotate(text=label, xy=(x, y), textcoords='offset points',
                    fontsize=150, ha='left', va='baseline')

    # Circular line:
    r = np.zeros(shape=(200))
    theta = 2 * np.pi * np.arange(0, 1, 0.005)
    r.fill(1)
    ax.scatter(theta, r, s=200, c='black')

    ax.set_title("Comorbidome", fontsize=300)
    plt.savefig(input_path.joinpath('graphs', 'Comorbidome.png'))
    plt.show()


comoridome_polar_plot(df_comorbidome_plot)

# Convert days to year, months, days format:


def convert_days(number_of_days):
    # Assume that years is of 365 days
    year = int(number_of_days / 365)
    month = int((number_of_days % 365) / 30)
    days = (number_of_days % 365) % 30
    return f'{year} years {month} months {days} days'


# Driver Code
number_of_days = 365
convert_days(number_of_days)

# Parameters fine tuning by grid search:
