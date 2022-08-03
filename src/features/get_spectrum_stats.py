#!/usr/bin/env python
import sys
import pickle#!/usr/bin/env python
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error

from tqdm import tqdm_notebook as tqdm
sys.path.append("../../")
import os
code_path = os.path.join(os.environ["HOME"], "code/github/ABP_pred")
sys.path.append(code_path)
from src.utils import *
from src.generator import DataGenerator
from src.callbacks import OutputPredictionCallback, TrainingPlot
import src.project_configs as project_configs
import src.models.abp_model as abp_model

num_coeffs = 50
num_iters = 100

load_scaler_pickle = True
load_weights = True

fig_file_string = "epoch_{}_pred_img_train_{}_{}.png"


if not load_scaler_pickle:
    train_gen = DataGenerator(data_dir=project_configs.train_dir, window_len=project_configs.window_size,
                              batch_size=project_configs.batch_size, use_spectrum_filtering=False)
    pickle.dump(train_gen.X_scaler, open("../../models/train_X_scaler.pkl", "wb"))
    pickle.dump(train_gen.y_scaler, open("../../models/train_y_scaler.pkl", "wb"))
else:
    X_scaler = pickle.load(open(os.path.join(code_path, "models/train_X_scaler.pkl"), "rb"))
    y_scaler = pickle.load(open(os.path.join(code_path, "models/train_y_scaler.pkl"), "rb"))
    train_gen = DataGenerator(data_dir=project_configs.train_dir,
                              window_len=project_configs.window_size,
                              batch_size=project_configs.batch_size,
                              X_scaler=X_scaler,
                              y_scaler=y_scaler,
                              use_spectrum_filtering=False)

# use mean/stdev from training data to scale testing data
val_gen = DataGenerator(data_dir=project_configs.test_dir,
                        window_len=project_configs.window_size,
                        batch_size=project_configs.batch_size,
                        X_scaler=train_gen.X_scaler,
                        y_scaler=train_gen.y_scaler,
                        use_spectrum_filtering=False)

spectrum_array = np.array([])
for i in tqdm(range(num_iters)):
    X, y = train_gen.__getitem__(i)
    for j in range(X.shape[0]):
        ppg = X[j, :, 5]
        P = np.correlate(ppg, ppg, mode="full")
        Q = np.fft.rfft(P).real[0:num_coeffs].reshape(-1, 1).T
        if spectrum_array.shape[0] == 0:
            spectrum_array = np.array(Q)
        else:
            spectrum_array = np.append(spectrum_array, Q, axis=0)

np.save("spectrum_array.npy", spectrum_array)

mean_spectrum = spectrum_array.mean(axis=0)
np.save("mean_spectrum.npy", mean_spectrum)

median_spectrum = np.median(spectrum_array, axis=0)
np.save("median_spectrum.npy", median_spectrum)

spectrum_diffs = []
for i in tqdm(range(num_iters)):
    X, y = train_gen.__getitem__(i)
    for j in range(X.shape[0]):
        ppg = X[j, :, 5]
        P = np.correlate(ppg, ppg, mode="full")
        Q = np.fft.rfft(P).real[0:num_coeffs].T
        mse = np.sum(np.abs(np.subtract(median_spectrum, Q)))
        spectrum_diffs.append(mse)

np.save("spectrum_diffs.npy", spectrum_diffs)