# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 12:29:04 2023

@author: may7e
"""
import pandas as pd
import os


from getKPindex import getKpindex



def binary_classification( datatype: str = 'Hp30', start_time: str = '2016-01-01',
                          stop_time: str = '2023-10-06', threshold: int = 3, save_path: str = '../Hp_Data/'):
    
    #Get the data 
    
    data = getKpindex('2016-01-01','2023-10-06',index = datatype)
    
    data = {'time': data[0], 'values': data[1]} 
    
    df = pd.DataFrame(data)
    
    
    #Create new columns for classifying whether the values go beyond a certain threshold
    df[f'above_{threshold}'] = (df['values'] > threshold).astype(int)
    df[f'below_{threshold}'] = (df['values'] < threshold).astype(int)
    
    #get the time to be in datetime 
    df['time'] = pd.to_datetime(df['time'])
    
    
    #Create folder in parent directory if it doens't exits
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    
    filename = os.path.join(save_path, 'binary_data.csv')
    df.to_csv(filename)
    
    
def multiple_classification(datatype: str = 'Hp30', start_time: str = '2016-01-01', 
                            stop_time: str = '2023-10-06', threshold: tuple = (3,6,9),save_path: str = '../Hp_Data/'):
    
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
    
    
    #Create folder in parent directory if it doens't exits
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    
    filename = os.path.join(save_path, 'multiclass_data.csv')
    df.to_csv(filename)
    
    
    
if __name__ == "__main__":
    print('Executing both functions')
    
    binary_classification()
    multiple_classification()
    
    print('DONE!!!')
    
    
    
    
    
    
    
    
    