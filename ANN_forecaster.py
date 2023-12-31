import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam
import utils
import preprocessing
from gap_filling import GapFiller
import tqdm
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # Suppress pesky performance warnings


def create_delays(df, name, time=20):
    '''
    For creating time history of a multivariate time series
    :param df: The input dataframe
    :param name: The list of names of the columns to create time history for
    :param time: The length of the time history window to create (this will create that many extra features)
    :return: None, the columns are added inplace
    '''
    for delay in np.arange(1, int(time) + 1):
        df[name + '_%s' % delay] = df[name].shift(delay).astype('float32')


# Fully-connected neural network class
class Forecaster(Sequential):
    def __init__(self, input_shape, output_shape, config):
        super().__init__()
        self.config = config
        self.input_shape_ = input_shape
        self.output_shape_ = output_shape
        self.build_model()

    def build_model(self):
        self.add(Dense(40, input_shape=self.input_shape_, activation="relu"))
        self.add(Dropout(0.2))
        self.add(Dense(20, activation="relu"))
        self.add(Dropout(0.2))
        self.add(Dense(10, activation="linear"))
        self.add(Dropout(0.2))
        self.add(Dense(self.output_shape_))


    def compile_model(self):
        self.compile(
            loss=self.config["loss"],
            optimizer=self.config["optimizer"]
        )

    def fit_model(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        self.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=10,
                    verbose=1,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                ),
                WandbMetricsLogger()
            ],
        )

    def evaluate_model(self, X_test, y_test, plot=True):
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if plot:
            plt.plot(range(len(y_pred[0])), y_pred, label="Predicted")
            plt.plot(range(len(y_test[0])), y_test, label="Actual")
            plt.title("Whether Hp30 is above 3")
            plt.legend()
            # plt.savefig("forecasting_performance.png")
            plt.show()
        return y_pred, rmse, mae, r2


def fill_gaps(df, linear_limit=0, ml_limit=0):
    '''
    Given a gap in the data, fill it using the gap filling model in `gap_filling.py`
    :param timestep: The input timestep to fill, including BX, BY, and BZ
    :return: The timestep with gaps filled by interpolation and ML
    '''
    gap_filler = GapFiller((3,), len(df.columns)-3,
                           config={"loss": "mse", "optimizer": Adam(learning_rate=1e-2)})
    gap_filler.load_weights("gap_filler.h5")

    # Interpolate small gaps linearly
    df.interpolate(method="linear", limit=linear_limit, inplace=True)

    for timestep in tqdm.trange(len(df), desc="Using ML to fill large gaps"):
        if df.iloc[timestep:timestep+ml_limit].isna().sum().sum():
            if df.iloc[timestep].isna().sum() and not df[["BX", "BY", "BZ"]].iloc[timestep].isna().sum():  # If there is a gap in the plasma data but not in the B-field
                atomic_prediction = gap_filler.predict(df[["BX", "BY", "BZ"]].iloc[timestep].values.reshape(1, 3), verbose=0)
                df.iloc[timestep] = pd.concat([df[["BX", "BY", "BZ"]].iloc[timestep],
                                               pd.DataFrame(np.reshape(atomic_prediction, (8,1)))], axis=0).T

    return gap_filler.predict(timestep)

if __name__ == "__main__":
    wandb.init(project="SpaceApp2023")

    use_pretrained_model = False
    load_filled_gaps = True

    # Load data
    dscovr_df = utils.read_all_data("data/dscovr")
    dscovr_time = dscovr_df["time"]
    dscovr_df.drop(columns=["time"], inplace=True)

    hp_df = pd.read_csv("multiclass_data.csv")
    hp_df = hp_df[["time", "above_3", "above_6", "above_9"]]
    hp_df["time"] = pd.to_datetime(hp_df["time"])

    # Preprocessing
    print("Preprocessing data...")
    preprocessor = preprocessing.dscovr_preprocessor()
    dscovr_df = preprocessor.drop_nan_threshold(dscovr_df, threshold=0.2)  # Drop features with more than 20% NaNs
    dscovr_df = preprocessor.select_across_k_cups(dscovr_df, k=5)  # Only use every k-th cup
    dscovr_df.values[:, 3:] = np.sqrt(dscovr_df.values[:, 3:])  # Take square root of all features that are not B-field

    if load_filled_gaps:
        print(dscovr_df.info())
        dscovr_df = fill_gaps(dscovr_df, linear_limit=10, ml_limit=30)
        dscovr_df.to_csv("filled_gaps.csv")
        print(dscovr_df.info())
    else:
        dscovr_df = pd.read_csv("filled_gaps.csv")


    for param in tqdm.tqdm(list(dscovr_df), desc=f"Creating time history"):
        create_delays(dscovr_df, param, time=30)

    dscovr_df["time"] = dscovr_time.dt.tz_localize("UTC")
    all_data = pd.merge(dscovr_df, hp_df, on="time", how="inner")

    # Drop NaNs
    original_len = len(all_data)
    print("Dropping Nans. Dataset size before: ", original_len)
    all_data.dropna(inplace=True, axis=0)
    print("Dataset size after: ", len(all_data))
    print(f"Percentage of data remaining: {len(all_data)/original_len*100:.2f}")

    print("Splitting and scaling")
    X = all_data[list(dscovr_df)]
    y = all_data[list(hp_df)]
    X.set_index("time", inplace=True, drop=True)
    y.set_index("time", inplace=True, drop=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True)
    X_scaler = StandardScaler()
    train_X = X_scaler.fit_transform(train_X)
    test_X = X_scaler.transform(test_X)

    if use_pretrained_model:
        # Since we're using a subclassed model, we need to re-instantiate it and load the weights into it
        forecaster = Forecaster((train_X.shape[1],), train_y.shape[1],
                               config={"loss": "mse", "optimizer": Adam(learning_rate=1e-3)})
        forecaster.load_weights("forecaster.h5")
    else:
        # Instantiate and train model
        forecaster = Forecaster((train_X.shape[1],), train_y.shape[1],
                               config={"loss": "mse", "optimizer": Adam(learning_rate=1e-3)})
        forecaster.compile_model()
        forecaster.fit_model(train_X, train_y, test_X, test_y, epochs=50, batch_size=64)

    predictions, rmse, mae, r2 = forecaster.evaluate_model(test_X, test_y, plot=True)
    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    # Save predictions and model
    predictions = pd.DataFrame(predictions, columns=list(hp_df))
    predictions.set_index(test_y.index, inplace=True)
    pd.DataFrame(predictions).to_csv("forecast_data.csv")
    forecaster.save_weights("forecaster.h5")

    print("Done! Yay!")
