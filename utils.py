#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import os



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