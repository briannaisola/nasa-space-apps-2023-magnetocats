####################################################################################
#
# exmining_twins_and_supermag/modeling_v0.py
#
# Performing the modeling using only the Solar Wind and Ground Magnetomoeter data.
# Similar model to Coughlan (2023) but with a different target variable. Will be
# used for comparison to the model that incorperates the reduced TWINS data.
#
####################################################################################


# Importing the libraries
import datetime
import gc
import glob
import json
import os
import pickle
import subprocess
import time
import yaml

import keras
import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from scipy.special import expit, inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.backend import get_session
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_prep import DataPrep

os.environ["CDF_LIB"] = "~/CDF/lib"


# loading config and specific model config files. Using them as dictonaries
config_path = "config.yaml"
with open(config_path, 'r') as f:
	MODEL_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def getting_prepared_data():
	'''
	Calling the data prep class without the TWINS data for this version of the model.

	Returns:
		X_train (np.array): training inputs for the model
		X_val (np.array): validation inputs for the model
		X_test (np.array): testing inputs for the model
		y_train (np.array): training targets for the model
		y_val (np.array): validation targets for the model
		y_test (np.array): testing targets for the model

	'''

	prep = DataPrep(region_path, region_number, solarwind_path, supermag_dir_path, twins_times_path,
					rsd_path, random_seed)

	X_train, X_val, X_test, y_train, y_val, y_test = prep.do_full_data_prep(CONFIG)


	return X_train, X_val, X_test, y_train, y_val, y_test


def create_CNN_model(n_features, loss='mse', early_stop_patience=10):
	'''
	Initializing our model

	Args:
		n_features (int): number of input features into the model
		loss (str, optional): loss function to be uesd for training. Defaults to 'categorical_crossentropy'.
		early_stop_patience (int, optional): number of epochs the model will continue training once there
												is no longer val loss improvements. Defaults to 3.

	Returns:
		object: model configuration ready for training
		object: early stopping conditions
	'''


	model = Sequential()						# initalizing the model

	model.add(Conv2D(MODEL_CONFIG['filters'], 3, padding='same',
								activation='relu', input_shape=(MODEL_CONFIG['time_history'], n_features, 1)))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Conv2D(MODEL_CONFIG['filters']*2, 2, padding='same', activation='relu'))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Flatten())							# changes dimensions of model. Not sure exactly how this works yet but improves results
	model.add(Dense(MODEL_CONFIG['filters']*2, activation='relu'))		# Adding dense layers with dropout in between
	model.add(Dropout(0.2))
	model.add(Dense(MODEL_CONFIG['filters'], activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='linear'))
	opt = tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['initial_learning_rate'])		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=loss)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting


	return model, early_stop


def fit_CNN(model, X_train, X_val, y_train, y_val, early_stop):
	'''
	Performs the actual fitting of the model.

	Args:
		model (keras model): model as defined in the create_model function.
		xtrain (3D np.array): training data inputs
		xval (3D np.array): validation inputs
		ytrain (2D np.array): training target vectors
		yval (2D np.array): validation target vectors
		early_stop (keras early stopping dict): predefined early stopping function

	Returns:
		model: fit model ready for making predictions.
	'''

	if not os.path.exists(f'models/model_version_{version}.h5'):

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		# doing the training! Yay!
		model.fit(Xtrain, ytrain, validation_data=(Xval, yval), verbose=1,
					shuffle=True, epochs=MODEL_CONFIG['epochs'],  callbacks=[early_stop])

		# saving the model
		model.save(f'models/model_version_{version}.h5')

	else:
		# loading the model if it has already been trained.
		model = load_model(f'models/model_version_{version}.h5')				# loading the models if already trained


	return model


def making_predictions(model, X_test, y_test):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (keras model): trained model
		Xtest (np.array): testing inputs
		ytest (np.array): testing targets

	Returns:
		results_df (pd.DataFrame): dataframe with the testing targets and predictions

	'''

	# reshaping the input data for a single channel
	Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))			# reshpaing for one channel input

	# Checking for nans in the input data
	print('Test input Nans: '+str(np.isnan(Xtest).sum()))


	predicted = model.predict(Xtest, verbose=1)						# predicting on the testing input data
	predicted = tf.gather(predicted, [1], axis=1)					# grabbing the positive node
	predicted = predicted.numpy()									# turning to a numpy array
	predicted = pd.Series(predicted.reshape(len(predicted),))		# and then into a pd.series

	results_df = pd.DataFrame({'y_test':y_test,
					'predicted': predicted})						# and storing the results


	return results_df

def calculate_some_metrics(results_df):

	# This is where we will calclate some metrics for the model

	# Returning all of the metrics ina  dataframe.
	metrics = pd.DataFrame()

	return metrics


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	X_train, X_val, X_test, y_train, y_val, y_test = getting_prepared_data()

	# creating the model
	print('Initalizing model...')
	MODEL, early_stop = create_CNN_model(X_train.shape[2], loss=MODEL_CONFIG['loss'],
											early_stop_patience=MODEL_CONFIG['early_stop_patience'])

	# fitting the model
	print('Fitting model...')
	MODEL = fit_CNN(MODEL, X_train, X_val, y_train, y_val, early_stop)

	# making predictions
	print('Making predictions...')
	results_df = making_predictions(MODEL, X_test, y_test)

	# saving the results
	print('Saving results...')
	results_df.to_feather('outputs/non_twins_results.feather')

	# calculating some metrics
	print('Calculating metrics...')
	metrics = calculate_some_metrics(results_df)

	# saving the metrics
	print('Saving metrics...')
	metrics.to_feather('outputs/non_twins_metrics.feather')



if __name__ == '__main__':
	main()
	print('It ran. God job!')
