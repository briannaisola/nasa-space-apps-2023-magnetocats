#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils import rolling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def plot_rolling_envelope(data, columns=None, window_str=None, min_pts=None):
    """
    Plot the rolling mean and std-dev of columns of data
    
    Specify window size with window_str (e.g. '30min')
    
    Specify minimal number / fraction of pts per time interval needed with min_pts
    (e.g. 15 for 15 mins out of 30, or 0.5)

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    columns : TYPE, optional
        DESCRIPTION. The default is None.
    window_str : TYPE, optional
        DESCRIPTION. The default is None.
    min_pts : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if columns is None:
        columns = np.setdiff1d( list(data),['time'] ).tolist()
    if isinstance(columns,str): columns = [ columns ]
    
    # compute rolling mean and std dev for columns
    roll_data = rolling(data,
                        columns=columns,
                        window_str=window_str,
                        method='mean',
                        min_pts=min_pts)
    roll_data = rolling(roll_data,
                        columns=columns,
                        window_str=window_str,
                        method='std',
                        min_pts=min_pts)
    
    fig, axes = plt.subplots(len(columns),1)
    if len(columns) == 1:
        axes = np.array([axes])
    
    times = data['time']
    for i, label in enumerate(columns):
        axes[i].plot( times, data[label], c='blue', lw=1 )
        
        mean = roll_data[label+'_mean']
        std = roll_data[label+'_std']
        
        ls_ = 'solid'
        
        # plot mean and +/- std dev
        axes[i].plot( times, mean, ls=ls_, c='red', lw=1 )
        axes[i].plot( times, mean+std, ls=ls_, c='black', lw=1 )
        axes[i].plot( times, mean-std, ls=ls_, c='black', lw=1 )
        
        if i < len(columns)-1:
            axes[i].tick_params(axis='x', labelbottom=False)
        else:
            axes[i].tick_params(axis='x', labelrotation=20)
            
            
            

def plot_var_overview(ax, t, var, ylab=False, **kwargs):
    """quick and dirty method for plotting overview of data

    Args:
        ax (plot axis): ax to plot on
        t (dataframe): time variable df
        var (dataframe): variable df to plot
        ylab (bool, optional): label y-axis. Defaults to False.
    """
    
    var_name = var.name # get variable name

    ax.plot(t,var, linewidth=1, label=var_name, **kwargs)
    if ylab == True:
        ax.set_xlabel(f"year")
    
    # plot legend
    ax.legend(bbox_to_anchor=(1, 1))
    
    
    