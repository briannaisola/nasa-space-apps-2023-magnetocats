# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 12:29:04 2023

@author: may7e
"""
import pandas as pd
import os


from getKPindex import getKpindex



def storm_conditions(data_path: str, time_frame: int = 1, unit: str = 'd'):
    '''
    

    Parameters
    ----------
    data_path : str
        DESCRIPTION.
    time_frame : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    '''
    df = pd.read_csv(data_path)
    
    #convert to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    df['time_stop'] = df['time'] + pd.to_timedelta(time_frame, unit = unit)
    
    df['time_start'] = df['time'] - pd.to_timedelta(time_frame, unit = unit)
    
        
    column_data = df.pop('Symh')
    
    # Insert the column at the desired index
    df.insert(4, 'Symh', column_data)
    
    
    selected_columns = ['time_start', 'time_stop']
    time_condition = list(df[selected_columns].itertuples(index=False, name=None))
    
    
    return time_condition
    



def binary_classification( datatype: str = 'Hp30', start_time: str = '2016-01-01',
                          stop_time: str = '2023-10-06', threshold: int = 3, save_path: str = '../Hp_Data/'):
    '''
    

    Parameters
    ----------
    datatype : str, optional
        DESCRIPTION. The default is 'Hp30'.
    start_time : str, optional
        DESCRIPTION. The default is '2016-01-01'.
    stop_time : str, optional
        DESCRIPTION. The default is '2023-10-06'.
    threshold : int, optional
        DESCRIPTION. The default is 3.
    save_path : str, optional
        DESCRIPTION. The default is '../Hp_Data/'.

    Returns
    -------
    None.

    '''
    
    #Get the data 
    
    data = getKpindex('2016-01-01','2023-10-06',index = datatype)
    
    data = {'time': data[0], 'values': data[1]} 
    
    df = pd.DataFrame(data)
    
    
    #Create new columns for classifying whether the values go beyond a certain threshold
    df[f'above_{threshold}'] = (df['values'] > threshold).astype(int)
    df[f'below_{threshold}'] = (df['values'] < threshold).astype(int)
    
    #get the time to be in datetime 
    df['time'] = pd.to_datetime(df['time'])
    
    #get the storm data
    storm_path = os.path.join(save_path, 'storm_data.csv')
    
    conditions = storm_conditions(storm_path)
    
    #initialize an empty column for storing the storm filtered data
    storm_df = pd.DataFrame(columns=df.columns)
    
    
    for start_time, end_time in conditions:
        temp_df = df[(df['time'] >= start_time.tz_localize('UTC')) & (df['time'] <= end_time.tz_localize('UTC'))]
        storm_df = pd.concat([storm_df, temp_df], ignore_index=True)
    
    #Create folder in parent directory if it doens't exits
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    
    filename = os.path.join(save_path, 'storm_binary_data.csv')
    storm_df.to_csv(filename)
    
    
def multiple_classification(datatype: str = 'Hp30', start_time: str = '2016-01-01', 
                            stop_time: str = '2023-10-06', threshold: tuple = (3,6,9),save_path: str = '../Hp_Data/'):
    '''
    

    Parameters
    ----------
    datatype : str, optional
        DESCRIPTION. The default is 'Hp30'.
    start_time : str, optional
        DESCRIPTION. The default is '2016-01-01'.
    stop_time : str, optional
        DESCRIPTION. The default is '2023-10-06'.
    threshold : tuple, optional
        DESCRIPTION. The default is (3,6,9).
    save_path : str, optional
        DESCRIPTION. The default is '../Hp_Data/'.

    Returns
    -------
    None.

    '''
    
    #Get the data 
    data = getKpindex('2016-01-01','2023-10-06',index = datatype)
    
    data = {'time': data[0], 'values': data[1]} 
    
    df = pd.DataFrame(data) 
    
    #define the thresholds 
    t1, t2, t3 = threshold
    
    
    #Create new columns for classifying whether the values go beyond a certain threshold
    df[f'above_{t1}'] = (df['values'] > t1).astype(int)
    df[f'below_{t1}'] = (df['values'] < t1).astype(int)
    
    df[f'above_{t2}'] = (df['values'] > t2).astype(int)
    df[f'below_{t2}'] = (df['values'] < t2).astype(int)
    
    df[f'above_{t3}'] = (df['values'] > t3).astype(int)
    df[f'below_{t3}'] = (df['values'] < t3).astype(int)
    
    #get the time to be in datetime 
    df['time'] = pd.to_datetime(df['time'])
    
    #get the storm data
    storm_path = os.path.join(save_path, 'storm_data.csv')
    
    conditions = storm_conditions(storm_path)
    
    #initialize an empty column for storing the storm filtered data
    storm_df = pd.DataFrame(columns=df.columns)
    
    
    for start_time, end_time in conditions:
        temp_df = df[(df['time'] >= start_time.tz_localize('UTC')) & (df['time'] <= end_time.tz_localize('UTC'))]
        storm_df = pd.concat([storm_df, temp_df], ignore_index=True)
    
    
    
    #Create folder in parent directory if it doens't exits
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    
    filename = os.path.join(save_path, 'storm_multiclass_data.csv')
    storm_df.to_csv(filename)
    
    
    
if __name__ == "__main__":
    print('Executing both functions')
    
    binary_classification()
    multiple_classification()
    
    print('DONE!!!')
    
    
    
    
    
    
    
    
    