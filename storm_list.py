# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:56:09 2023

@author: may7e
"""


import os
import pandas as pd
import spacepy.pycdf as cdf


def storm_list(path, threshold: int = -50, min_spacing: int = 2, freq: int = 5, save_path: str = '../Hp_Data/'):
    '''
    

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    threshold : int, optional
        DESCRIPTION. The default is 50.
    min_spacing : int, optional
        DESCRIPTION. The default is 1.
    freq : int, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    '''
    #calculate the minimum time between the detection of one event and the next.
    min_spacing = min_spacing*(1440/freq)

    storm_time = []
    storm_value = []
    
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path,folder)):
            
            data = cdf.CDF(os.path.join(path,folder,file))
            
            time = data['Epoch'][:]
            symh = data['SYM_H'][:]
        
        
            prev_value = None
            prev_index = None
            
            
            for index, value in enumerate(symh):
                if value <= threshold:
                    
                    # Check if it's the first selected value or if it's sufficiently spaced apart
                    if prev_value is None or (index - prev_index) >= min_spacing:
                        
                        epoch = time[index]
                        # Append the value to the selected_values list
                        storm_value.append(value)
                        storm_time.append(epoch)
                        
                        # Update the previous selected value and its index
                        prev_value = value
                        prev_index = index
    
    data = {'time': storm_time, 'Symh': storm_value}
    
    df = pd.DataFrame(data)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
       print(f"Folder '{save_path}' already exists.")
    
    
    filename = os.path.join(save_path, 'storm_data.csv')
    df.to_csv(filename)
    
if __name__ == "__main__":
    print('Getting Storm Data')
    
    path = 'D:/OMNI/'
    
    storm_list(path = path)
    
    print('DONE!!!')
    
    
    
    
    
    
    
    