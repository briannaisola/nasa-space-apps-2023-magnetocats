import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import utils


# Fully-connected neural network class
class GapFiller(Sequential):
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
            optimizer=self.config["optimizer"],
            metrics=self.config["metrics"],
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
                    patience=25,
                    verbose=1,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                )
            ],
        )

    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return rmse, mae, r2

    def save_model(self, model_path):
        self.save(model_path)

if __name__ == "__main__":
    wandb.init(project="SpaceApp2023")

    dscovr_df = utils.read_all_data("data/dscovr")
    dscovr_df.set_index("time", inplace=True, drop=True)

    original_len = len(dscovr_df)
    print("Dropping Nans. Dataset size before: ", original_len)
    dscovr_df.dropna(inplace=True, axis=0)
    print("Dataset size after: ", len(dscovr_df))
    print(f"Percentage of data remaining: {len(dscovr_df)/original_len:.2f}")

    print("Splitting and scaling")
    magnetic_df = dscovr_df[["BX", "BY", "BZ"]]
    plasma_df = dscovr_df.drop(["BX", "BY", "BZ"], axis=1)
    train_X, test_X, train_y, test_y = train_test_split(magnetic_df, plasma_df, test_size=0.2, shuffle=True)

    gap_filler = GapFiller((train_X.shape[1],), train_y.shape[1],
                           config={"loss": "mse", "optimizer": "adam", "metrics": ["mse"]})
    gap_filler.compile_model()
    gap_filler.fit_model(train_X, train_y, test_X, test_y, epochs=100, batch_size=32)

    rmse, mae, r2 = gap_filler.evaluate_model(test_X, test_y)
    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
    print("Done! Yay!")
