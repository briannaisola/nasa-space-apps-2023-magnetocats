####################################################################################
#
# nasa-space-apps-2023-magnetocats/hyperparameter_tuning.py
#
# Doing hyperparameter tuning for the CNN model using optuna package.
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

import keras
import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly
import shapely
import tensorflow as tf
import tqdm
from optuna_dashboard import run_server
from scipy.special import expit, inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from spacepy import pycdf
from tensorflow.keras.backend import clear_session, int_shape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten,
                                     Input, MaxPooling2D, Reshape, concatenate)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.backend import get_session

from data_generator import Generator
from data_prep import DataPrep

os.environ["CDF_LIB"] = "~/CDF/lib"

input_file = ''
target_file = ''
random_seed = 42

# loading config and specific model config files. Using them as dictonaries
with open('hyperparameter_tuning_config.json', 'r') as con:
	CONFIG = json.load(con)


def getting_prepared_data():
	'''
	Calling the data prep class without the TWINS data for this version of the model.

	Returns:
		Xtrain (np.array): training inputs for the model
		Xval (np.array): validation inputs for the model
		Xtest (np.array): testing inputs for the model
		ytrain (np.array): training targets for the model
		yval (np.array): validation targets for the model
		ytest (np.array): testing targets for the model

	'''

	prep = DataPrep(region_path, region_number, solarwind_path, supermag_dir_path, twins_times_path,
					rsd_path, random_seed)

	Xtrain, Xval, Xtest, ytrain, yval, ytest  = prep.full_data_prep(DATA_PREP_CONFIG)

	Xtrain = train[:(int(len(train)*0.1)),:,:]
	ytrain = train[:(int(len(train)*0.1)),:,:]
	Xval = val[:(int(len(val)*0.1)),:,:]
	yval = val[:(int(len(val)*0.1)),:,:]

	print(f'Xtrain shape: {Xtrain.shape}')
	print(f'ytrain shape: {ytrain.shape}')
	print(f'Xval shape: {Xval.shape}')
	print(f'yval shape: {yval.shape}')

	# reshaping the model input vectors for a single channel
	Xtrain = Xtrain.reshape((Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1))
	Xval = Xval.reshape((Xval.shape[0], Xval.shape[1], Xval.shape[2], 1))
	Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))

	input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)

	# train = Generator(train, train, batch_size=16, shuffle=True)
	# val = Generator(val, val, batch_size=16, shuffle=True)

	return Xtrain, Xval, Xtest, ytrain, yval, ytest, input_shape


def CNN(input_shape, trial, early_stopping_patience=10):


	initial_filters = trial.suggest_categorical('initial_filters', [32, 64, 128])
	learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-2)
	cnn_layers = trial.suggest_int('layers', 2, 4)
	dense_layers = trial.suggest_int('dense_layers', 2, 4)
	maxpool = trial.suggest_categorical('maxpool', [True, False])
	activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])

	loss_function = CONFIG['loss_function']

	print(f'YOURE USING {loss_function} LOSS FUNCTION!!!!')

	output_dim = 2
	output_activation = 'softmax'

	model_input = Input(shape=input_shape)
	filters = initial_filters

	for i in range(cnn_layers):

		if i == 0:
			c = Conv2D(filters=filters, kernel_size=3, activation=activation, strides=1, padding='same')(model_input)
			if maxpool:
				c = MaxPooling2D()(c)
			filters = (filters*2)
		else:
			c = Conv2D(filters=filters, kernel_size=3, activation=activation, strides=1, padding='same')(c)
			if maxpool:
				c = MaxPooling2D()(c)
			filters = (filters*2)

	shape = int_shape(c)

	c = Flatten()(c)

	first_dense_nodes = shape[1]*shape[2]*shape[3]

	d = Dense(first_dense_nodes)(d)

	for i in range(dense_layers):
		first_dense_nodes = int(first_dense_nodes/2)
		d = Dense(first_dense_nodes)(d)

	output_layer = Dense(output_dim, activation=output_activation)(d)

	full_model = Model(inputs=model_input, outputs=model_outputs)

	opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)		# learning rate that actually started producing good results
	full_model.compile(optimizer=opt, loss=loss_function)		# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)		# early stop process prevents overfitting

	return full_model, early_stop


def objective(trial, Xtrain, Xval, Xtest, ytrain, yval, ytest, input_shape):

	model, early_stop = CNN(input_shape, trial)
	print(model.summary())
	clear_session()
	try:
		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
				verbose=1, shuffle=True, epochs=500,
				callbacks=[early_stop], batch_size=CONFIG['batch_size'])			# doing the training! Yay!
	except:
		print('Resource Exhausted Error')
		return None

	return model.evaluate(Xtest, ytest, verbose=1)


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''

	# loading all data and indicies
	print('Loading data...')
	Xtrain, Xval, Xtest, ytrain, yval, ytest, input_shape = getting_prepared_data()

	storage = optuna.storages.InMemoryStorage()
	# reshaping the model input vectors for a single channel
	study = optuna.create_study(direction='minimize', study_name='nasa_cnn', storage=storage, load_if_exists=True)
	study.optimize(lambda trial: objective(trial, Xtrain, Xval, Xtest, ytrain, yval, ytest, input_shape), n_trials=10, callbacks=[lambda study, trial: gc.collect()])
	print(study.best_params)

	run_server(storage)

	optuna.visualization.plot_param_importances(study).write_image('plots/param_importances.png')

	best_model, ___ = Autoencoder(input_shape, study.best_params)

	best_model.evaluate(Xtest, ytest)

	best_model.save('models/best_cnns.h5')

	optuna.visualization.plot_optimization_history(study).write_image('plots/optimization_history.png')



if __name__ == '__main__':
	main()
	print('It ran. God job!')
