####################################################################################
#
# nasa-space-apps-2023-magnetocats/data_prep.py
#
# The file contains teh class for preparing the data for the modeling. This process uses
# solar wind data from the DscovR satellite
#
####################################################################################


# Importing the libraries

import glob
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf

os.environ["CDF_LIB"] = "~/CDF/lib"

pd.options.mode.chained_assignment = None  # default='warn'


class DataPrep:

	def __init__(self, input_files, target_files, time_history, random_seed):
		'''
		Initialization function for the class.

		Args:
			input_files (list): list of input files
			target_files (list): list of target files
			time_history (int): number of minutes of solar wind data to include in the dataframe
			random_seed (int): random seed for reproducibility
		'''

		self.input_files = input_files
		self.target_files = target_files
		self.time_history = time_history
		self.random_seed = random_seed


	def loading_global_data(self):
		'''
		Loads the global data for the modeling.

		Args:
			load_twins (bool): whether or not to load the TWINS data
			twins_dir (string): path to the TWINS data. Required if load_twins == True
			twins_col_limits (list): column limits for the TWINS data with the first featent the min and the second the max
			twins_row_limits (list): row limits for the TWINS data with the first featent the min and the second the max
		'''

		# loading the solar wind data and setting datetime index
		self.input_data = pd.read_feather(self.input_files)
		self.input_data.set_index('Epoch', inplace=True, drop=True)
		self.input_data.index = pd.to_datetime(self.input_data.index)

		# loading the solar wind data and setting datetime index
		self.target_data = pd.read_feather(self.target_files)
		self.target_data.set_index('Epoch', inplace=True, drop=True)
		self.target_data.index = pd.to_datetime(self.target_data.index)


	def processing_solarwind_data(self, features, rolling=False, rolling_window=None,
								rolling_vars='all', data_manipulations=None, to_drop=None):
		'''
		Combines the regional dataframes into one dataframe for modeling.

		Args:
			features (list): list of features to include in the dataframe
			rolling (bool): whether or not to apply a rolling window to the dataframe. Applies maximum value to the window.
							Only used if rolling == True. Uses forward rolling indexer. Will need to be re-written to inclue
							backward rolling indexer and use of the mean instead of the max value in the window.
			rolling_window (int): number of minutes to use for the rolling window. Only used if rolling == True.
			rolling_vars (list): list of columns to apply the rolling window to. Only used if rolling == True.
			data_manipulations (string or list of strings): list of data manipulations to apply to the dataframe such as
								"mean", "std", "max" etc.
			to_drop (list): list of columns to drop from the dataframe.

		Returns:
			pd.DataFrame: dataframe containing the regional data
		'''

		if rolling:
			if rolling_vars == 'all':
				rolling_vars = self.input_data.columns

			if rolling_window is None:
				raise ValueError('Must specify a rolling window size.')

			if data_manipulations is None:
				raise ValueError('Must specify a valid data manipulation to apply to the rolling window. Options are "mean", "std", and "max".')

			for var in rolling_vars:
				if data_manipulations == 'max':
					self.input_data[f'{var}_rolling_{data_manipulations}'] = self.input_data[var].rolling(window=rolling_window, min_periods=1).max()
				elif data_manipulations == 'mean':
					self.input_data[f'{var}_rolling_{data_manipulations}'] = self.input_data[var].rolling(window=rolling_window, min_periods=1).mean()
				elif data_manipulations == 'std':
					self.input_data[f'{var}_rolling_{data_manipulations}'] = self.input_data[var].rolling(window=rolling_window, min_periods=1).std()

		if to_drop is not None:
			self.input_data.drop(to_drop, axis=1, inplace=True)

		return self.input_data


	def split_sequences(self, sequences, target, n_steps=30):
		'''
			Takes input from the input array and creates the input and target arrays that can go into the models.

			Args:
				sequences (np.array): input features. Shape = (length of data, number of input features)
				results_y: series data of the targets for each threshold. Shape = (length of data, 1)
				n_steps (int): the time history that will define the 2nd demension of the resulting array.
				include_target (bool): true if there will be a target output. False for the testing data.

			Returns:
				np.array (n, time history, n_features): array for model input
				np.array (n, 1): target array
			'''

		# Getting the index values of the df based on maching the
		# Date_UTC column and the value fo the twins_dates series. '''

		X, y1 = list(), list()		# creating lists for storing results
		for i in range(n_steps, len(sequences)):								# going to the end of the dataframes
			beginning_ix = i - n_steps					# find the end of this pattern
			if beginning_ix < 0:					# check if we are beyond the dataset
				raise ValueError('Time history goes below the beginning of the dataset')
			seq_x = sequences[beginning_ix:i, :].to_numpy()				# grabs the appropriate chunk of the data

			if np.isnan(seq_x).any():				# doesn't add arrays with nan values to the training set
				print(f'nan values in the input array for {df["Date_UTC"][i]}')
				bad_dates.append(df['Date_UTC'][i])
				continue
			seq_y1 = target[i]				# gets the appropriate target
			y1.append(seq_y1)
			X.append(seq_x)

		return np.array(X), np.array(y1)


	def splitting_and_scaling(self, scaling_method='standard', test_size=0.2, val_size=0.25):
		'''
		Splits the data into training, validation, and testing sets and scales the data.

		Args:
			scaling_method (string): scaling method to use for the solar wind and
															supermag data. Options are 'standard' and 'minmax'.
															Defaults to 'standard'.
			test_size (float): size of the testing set. Defaults to 0.2.
			val_size (float): size of the validation set. Defaults to 0.25. This equates to a 60-20-20 split for train-val-test

		Returns:
			np.array: training input array
			np.array: testing input array
			np.array: validation input array
			np.array: training target array
			np.array: testing target array
			np.array: validation target array
		'''
		self.X, self.y = self.split_sequences(supermag_and_solarwind_data, target_var=target_var, n_steps=time_history)

		self.Xtrain, self.twins_x_test = train_test_split(self.input_data, test_size=test_size, random_state=self.random_seed)
		self.Xtrain, self.Xval, self.ytrain, self.yval = train_test_split(self.Xtrain, self.ytrain, test_size=val_size, random_state=self.random_seed)

		# defining the scaler
		#IMPLIMENT WHATEVER SCALING ANDY FOUND TO BE BEST HERE!

		# scaling the TWINS data
		self.Xtrain = self.scaler.fit_transform(self.Xtrain)
		self.Xtest = self.scaler.transform(self.Xtest)
		self.Xval = self.scaler.transform(self.Xval)


		return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val


	def do_full_data_prep(self, config=None):

		if config is None:
			raise ValueError('Must specify a config file or variable dictionary.')

		# loading gloabl data
		self.loading_global_data()

		# processing solarwind data
		self.solarwind_data = self.processing_solarwind_data(features=config['sw_features'],
										rolling=config['sw_rolling'], rolling_window=config['sw_rolling_window'],
										rolling_vars=config['sw_rolling_vars'], data_manipulations=config['sw_data_manipulations'],
										to_drop=config['sw_to_drop'])

		# splitting and scaling the data
		self.splitting_and_scaling(scaling_method=config['scaling_method'], test_size=config['test_size'], val_size=config['val_size'])


		return self.Xtrain, self.Xval, self.Xtest, self.ytrain, self.yval, self.ytest
