#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import os
import tqdm




def read_all_data(data_folder):
    dat = []
    for year in tqdm.tqdm(np.arange(2016,2023+1)):
        dat.append( read_year(data_folder, year) )
    return pd.concat(dat)





def read_year(data_folder, year_int):
    file_name = [ elem for elem in os.listdir(data_folder) if str(year_int) in elem ][0]
    file_path = os.path.join(data_folder, file_name)
    data = pd.read_csv(file_path,
                       delimiter = ',',
                       parse_dates=[0],
                       infer_datetime_format=True, na_values='0',
                       header = None)
    columns = { 
        key : elem for key, elem in \
            zip( 
                  np.arange(data.shape[1]),
                  ['time','BX','BY','BZ', *[ 'FC'+str(elem) for elem in np.arange(data.shape[1]-4) ]]
                )
               }
    return data.rename(columns=columns)



def time_subset(data, start=None, end=None):

    if start is None: start = pd.to_datetime(data['time'].iloc[0], unit='s')
    if end is None: start = pd.to_datetime(data['time'].iloc[-1], unit='s')
    mask = (pd.to_datetime(data['time'], unit='s') >= start) \
           & (pd.to_datetime(data['time'], unit='s') <= end)
    return data[mask]




def rolling(data, method=None, window_str=None, columns=None, min_pts=None):
    """
    Get rolling average of columns of data, ignoring Nans
    
    window_str is pandas Timedelta of window size (e.g. '1hour' or '10min')
    
    min_pts is how many points should be in interval to compute; either int
    or frac (e.g. 0.5 means 50% of time interval should be occupied)
    """
    if method is None: method = 'mean'
    if window_str is None: window_str = '30min'
    if columns is None:
        columns = np.setdiff1d( list(data),['time'] ).tolist()
    if isinstance(columns,str): columns = [ columns ]
    if min_pts is None: min_pts = 1
    
    win_size = int( pd.Timedelta(window_str) / (data['time'].iloc[1] - data['time'].iloc[0]) )
    
    # if min_pts is float, then grab at least that fraction of data per window
    if min_pts < 1:
        min_pts = int(min_pts * win_size)
    
    rolling_ = data[columns].rolling(win_size, min_periods=min_pts)
    results = None
    if method == 'mean':
        results = rolling_.mean()
    elif method == 'std':
        results = rolling_.std()
    else:
        raise ValueError('Unrecognized method')
    
    old_columns = [ 'time', *columns ]
    new_columns = ['time', *[ elem+'_'+method for elem in columns ] ]
    new_labels = { key : elem for key,elem in zip(old_columns, new_columns) }
    results = results.rename(columns=new_labels)
    
    return pd.concat([data,results], axis=1)