# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 12:29:04 2023

@author: may7e
"""
import pandas as pd
import os

# Import the getKpindex function from getKPindex.py
from getKPindex import getKpindex

# Function to create time conditions for storm data
def storm_conditions(data_path: str, time_frame: int = 1, unit: str = 'd'):
    '''
    Parameters:
        data_path (str): Path to the storm data file.
        time_frame (int): Time frame for storm conditions.
        unit (str): Time unit for the time frame (e.g., 'd' for days).

    Returns:
        list: List of time conditions (start_time, end_time).
    '''
    
    # Read the storm data from the specified path
    df = pd.read_csv(data_path)
    
    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Calculate start and stop times based on the time frame
    df['time_stop'] = df['time'] + pd.to_timedelta(time_frame, unit=unit)
    df['time_start'] = df['time'] - pd.to_timedelta(time_frame, unit=unit)
    
    # Move 'Symh' column to the 4th position
    column_data = df.pop('Symh')
    df.insert(4, 'Symh', column_data)
    
    # Select only 'time_start' and 'time_stop' columns and create a list of time conditions
    selected_columns = ['time_start', 'time_stop']
    time_condition = list(df[selected_columns].itertuples(index=False, name=None))
    
    return time_condition

# Function for binary classification of Kp index data
def binary_classification(datatype: str = 'Hp30', start_time: str = '2016-01-01',
                          stop_time: str = '2023-10-06', threshold: int = 3, save_path: str = '../Hp_Data/'):
    '''
    Parameters:
        datatype (str): Type of Kp index data (e.g., 'Hp30').
        start_time (str): Start date for data retrieval.
        stop_time (str): Stop date for data retrieval.
        threshold (int): Threshold for binary classification.
        save_path (str): Path to save the resulting data.

    Returns:
        None
    '''
    
    # Get Kp index data using the getKpindex function
    data = getKpindex(start_time, stop_time, index=datatype)
    
    # Create a DataFrame from the retrieved data
    data = {'time': data[0], 'values': data[1]}
    df = pd.DataFrame(data)
    
    # Create new columns for binary classification based on the threshold
    df[f'above_{threshold}'] = (df['values'] > threshold).astype(int)
    df[f'below_{threshold}'] = (df['values'] < threshold).astype(int)
    
    # Convert the 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Get storm data conditions
    storm_path = os.path.join(save_path, 'storm_data.csv')
    conditions = storm_conditions(storm_path)
    
    # Initialize an empty DataFrame for storing the storm-filtered data
    storm_df = pd.DataFrame(columns=df.columns)
    
    # Iterate through storm conditions and filter the data
    for start_time, end_time in conditions:
        temp_df = df[(df['time'] >= start_time.tz_localize('UTC')) & (df['time'] <= end_time.tz_localize('UTC'))]
        storm_df = pd.concat([storm_df, temp_df], ignore_index=True)
    
    # Create a folder in the parent directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    # Save the storm-filtered data to a CSV file
    filename = os.path.join(save_path, 'storm_binary_data.csv')
    storm_df.to_csv(filename)

# Function for multiclass classification of Kp index data
def multiple_classification(datatype: str = 'Hp30', start_time: str = '2016-01-01', 
                            stop_time: str = '2023-10-06', threshold: tuple = (3, 6, 9), save_path: str = '../Hp_Data/'):
    '''
    Parameters:
        datatype (str): Type of Kp index data (e.g., 'Hp30').
        start_time (str): Start date for data retrieval.
        stop_time (str): Stop date for data retrieval.
        threshold (tuple): Thresholds for multiclass classification.
        save_path (str): Path to save the resulting data.

    Returns:
        None
    '''
    
    # Get Kp index data using the getKpindex function
    data = getKpindex(start_time, stop_time, index=datatype)
    
    # Create a DataFrame from the retrieved data
    data = {'time': data[0], 'values': data[1]}
    df = pd.DataFrame(data) 
    
    # Define the thresholds
    t1, t2, t3 = threshold
    
    # Create new columns for multiclass classification based on the thresholds
    df[f'above_{t1}'] = (df['values'] > t1).astype(int)
    df[f'below_{t1}'] = (df['values'] < t1).astype(int)
    
    df[f'above_{t2}'] = (df['values'] > t2).astype(int)
    df[f'below_{t2}'] = (df['values'] < t2).astype(int)
    
    df[f'above_{t3}'] = (df['values'] > t3).astype(int)
    df[f'below_{t3}'] = (df['values'] < t3).astype(int)
    
    # Convert the 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Get storm data conditions
    storm_path = os.path.join(save_path, 'storm_data.csv')
    conditions = storm_conditions(storm_path)
    
    # Initialize an empty DataFrame for storing the storm-filtered data
    storm_df = pd.DataFrame(columns=df.columns)
    
    # Iterate through storm conditions and filter the data
    for start_time, end_time in conditions:
        temp_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        storm_df = pd.concat([storm_df, temp_df], ignore_index=True)
    
    # Create a folder in the parent directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    # Save the storm-filtered data to a CSV file
    filename = os.path.join(save_path, 'storm_multiclass_data.csv')
    storm_df.to_csv(filename)

if __name__ == "__main__":
    print('Executing both functions')
    
    # Call the binary_classification and multiple_classification functions
    binary_classification()
    multiple_classification()
    
    print('DONE!!!')

    
    
    
    
    
    
    
    
    