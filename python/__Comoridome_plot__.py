#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:43:08 2023

@author: image_in
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#df_row = np.arange(0, 2, 0.1)
#prevalence = np.random.rand(1, df_row.shape[0])*100

def comoridome_polar_plot(df_):
    prev = df_['prevalence_%']
    hr = df_['log(hr)']
    p_sign = df_['p_sign']
    p_sign.loc[p_sign] = 'red'
    groups = df_['groups']

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    
    # Parameters
    ax.set_rmax(max(hr))
    ax.set_rticks([])  # Less radial ticks
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    palette = dict(zip(
        groups, sns.color_palette(palette="muted", 
                                           n_colors=len(groups))
        ))
    
    ## Plot:
    # Points:
    angle =  np.random.rand(1, hr.shape[0])
    #ax.scatter(angle, hr, s=prevalence, c=colors)
    for g in groups:
        idxs = df_.index[df_['groups']==g]
        ax.scatter(angle.loc[idxs], hr.loc[idxs], c=palette[g], 
                   label=g, s=prev, edgecolors=p_sign)
        ax.legend()
    for i, (label, _) in enumerate(hr.items()):
        ax.annotate(texte=label, xy=(angle[i], hr[i]))
    # Circular line:
    r = np.zeros(shape=(100))
    r.fill(1)
    theta_r = np.arange(0, 2, 0.02)
    theta = 2 * np.pi * theta_r
    r.fill(1)
    ax.scatter(theta, r, s=0.3)
    
    ax.set_title("Comorbidome", va='bottom')
    plt.show()
    
    
from matplotlib.pyplot import cm
import numpy as np

#variable n below should be number of curves to plot

#version 1:

color = cm.rainbow(np.linspace(0, 1, 10))

"""    
# Annotate points:
import matplotlib.pyplot as plt
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 123]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))"""