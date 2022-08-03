import os
import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer
import pickle
from tqdm import tqdm
import glob

from src.utils import get_art_peaks
from src.ppg_qi import train_ppg_qi
import src.project_configs as project_configs
import src.utils as utils


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, data_dir, window_len=300, batch_size=32, step_size=1, shuffle=False, X_scaler=None,
                 y_scaler=None, use_spectrum_filtering=False, use_ppg_qi=False, wave_or_scaler="wave"):
        """Initialization"""
        self.data_dir = data_dir
        self.window_len = window_len  # number of samples to use in window
        self.batch_size = batch_size  # number of windows to use in sliding window
        self.step_size = step_size  # number of samples between windows in sliding window
        self.shuffle = shuffle  # if shuffle, don't use sliding window for batches
        self.data_files = os.listdir(self.data_dir)
        self.num_windows = 0
        self.file_count = 0
        self.median_spectrum_file = os.path.join(os.environ["HOME"], "code/github/ABP_pred/models/median_spectrum.npy")
        self.spectrum_diffs_file = os.path.join(os.environ["HOME"], "code/github/ABP_pred/models/spectrum_diffs.npy")
        self.use_spectrum_filtering = use_spectrum_filtering
        self.ppg_qi_model_weights = os.path.join(os.environ["HOME"], "code/github/ABP_pred/models/ppg_qi_model.hdf5")
        self.ppg_qi_scaler_file = os.path.join(os.environ["HOME"], "code/github/ABP_pred/models/ppg_qi_scaler.pkl")
        self.use_ppg_qi = use_ppg_qi
        self.wave_or_scaler = wave_or_scaler  # should target to predict be waveform or scaler values?

        if self.use_ppg_qi:
            self.ppg_qi_model = train_ppg_qi.create_model()
            self.ppg_qi_model.load_weights(self.ppg_qi_model_weights)
            self.ppg_qi_model._make_predict_function()
            self.ppg_qi_scaler = pickle.load(open(self.ppg_qi_scaler_file, "rb"))

        if self.use_spectrum_filtering:
            # load median spectrum vector from file
            self.median_spectrum_array = np.load(self.median_spectrum_file)
            # load spectrum_diffs array
            self.spectrum_diffs_array = np.load(self.spectrum_diffs_file)
            # calculate 95th percentile
            self.spectrum_diff_cutoff = np.percentile(self.spectrum_diffs_array, q=95)
            print("Using {} as 95th percentile cutoff for error".format(self.spectrum_diff_cutoff))

        self.demo_df = utils.get_demo_df()
        self.demo_df.drop_duplicates(inplace=True)

        # if this is for training data, we need to fit scalers
        if X_scaler is None and y_scaler is None:
            print("initializing new StandardScaler objects")
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        # otherwise, for test we initialize to training scalers
        else:
            print("using supplied StandardScaler objects")
            self.X_scaler = X_scaler
            self.y_scaler = y_scaler
            return
        for f in tqdm(self.data_files):
            self.file_count += 1
            if self.file_count % 5000 == 0:
                pickle.dump(self.X_scaler, open("train_X_scaler.pkl", 'wb'))
                pickle.dump(self.y_scaler, open("train_y_scaler.pkl", 'wb'))
            #         for f in self.data_files:
            X = np.load(os.path.join(self.data_dir, f), allow_pickle=True)
            #             print(X.shape)
            num_windows = int(X.shape[0] / self.window_len)
            self.num_windows += num_windows
            if X_scaler is None and y_scaler is None and not np.isnan(np.var(X[:, 0:-1])):
                # change here for ECG + SpO2
                self.X_scaler.partial_fit(X[:, 0:-1])
                # last column is ABP
                self.y_scaler.partial_fit(X[:, -1].reshape(-1, 1))
            elif np.isnan(np.var(X[:, 0:-1])):
                print("WARNING: found nan variance in {}".format(f))
                # if we have nan variance, it is usually from NIBP-derived features
                X = np.nan_to_num(X)
                self.X_scaler.partial_fit(X[:, 0:-1])
                # last column is ABP
                self.y_scaler.partial_fit(X[:, -1].reshape(-1, 1))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # return max(int(np.floor(self.num_windows / self.batch_size)), 1)
        return max(int(np.floor(len(glob.glob(os.path.join(self.data_dir, "*.npy")))*project_configs.max_windows_per_file / self.batch_size)), 1)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Input matrix
        batch_X = list()
        # waveform target
        batch_y = list()

        #data_files = os.listdir(self.data_dir)
        data_files = self.data_files
        ## this is for reading data from a single patient per batch
        #         f = random.choice(data_files)
        #         X = np.load(os.path.join(self.data_dir, f))
        #         num_windows = int(X.shape[0] / self.window_len)
        #         for val in np.random.randint(low=0, high=num_windows, size=self.batch_size):
        # this is for reading one window per patient per batch
        while len(batch_y) != self.batch_size:
            for f in np.random.choice(data_files, size=1):
                X = np.load(os.path.join(self.data_dir, f), allow_pickle=True)
                num_windows = int(X.shape[0] / self.window_len)
                # choose a random spot in the waveform
                val = np.random.randint(low=0, high=num_windows)
                idx = int(val * self.window_len)
                # we use a sliding window to check if we have a valid batch of data
                # (i.e. every window in in sliding window needs to be valid; this possibly
                # can be relaxed using some threshold)
                xx = X[idx:idx + self.window_len, project_configs.ecg_col]
                yy = X[idx:idx + self.window_len, project_configs.ppg_col]
                zz = X[idx:idx + self.window_len, project_configs.abp_col]

                nibp_sys = X[idx:idx + self.window_len, project_configs.nibp_sys_col].mean()
                nibp_dias = X[idx:idx + self.window_len, project_configs.nibp_dias_col].mean()
                nibp_mean = X[idx:idx + self.window_len, project_configs.nibp_mean_col].mean()

                # Fix for build_features.py bug..
                if nibp_sys <= 0 or nibp_dias <= 0 or nibp_mean <= 0:
                    continue

                pad_size = 4
                # replace all NaNs with 0
                X = np.nan_to_num(X)
                # draw random noise value and add to pseudo-NIBP to simulate cuff error
                if project_configs.add_nibp_noise:
                    noise_val = np.random.normal(loc=11.8, scale=28.9, size=1)
                    X[idx:idx + self.window_len, -4:-1] += noise_val
                # change here for ECG + SpO2
                # scaled_ppg_index = project_configs.ppg_col
                # min_max_scaler = MinMaxScaler(feature_range=(nibp_dias, nibp_sys))
                # X[idx:idx + self.window_len, scaled_ppg_index] = min_max_scaler.fit_transform(yy.reshape(-1, 1))[:, 0]

                # get demographic data for this patient
                # TODO: see why some patients don't have clinical data
                try:
                    patient_id = utils.get_patient_from_file(f)
                    X_demo = np.repeat(np.unique(self.demo_df.loc[patient_id]), project_configs.window_size)
                    X_demo = X_demo.reshape(project_configs.window_size, self.demo_df.shape[1])
                except KeyError:
                    continue

                if self.use_spectrum_filtering:
                    # check to see if PPG signal is valid
                    P = np.correlate(yy, yy, mode="full")
                    Q = np.fft.rfft(P).real[0:self.median_spectrum_array.shape[0]].T
                    error = np.sum(np.abs(np.subtract(self.median_spectrum_array, Q)))

                if self.use_ppg_qi:
                    qi_threshold = 0.5
                    yy_scaled = self.ppg_qi_scaler.transform(yy.reshape(-1, 1))
                    ppg_qi_pred = self.ppg_qi_model.predict(np.array([yy_scaled, ]))[0][0]

                # if error is less than our cutoff, continue with window
                # if self.use_spectrum_filtering is False or error <= self.spectrum_diff_cutoff:
                if self.use_ppg_qi is False or ppg_qi_pred > qi_threshold:
                    # get indices of systolic and diastolic BP
                    # bp_max_indices, bp_min_indices = get_art_peaks(y_scaler.inverse_transform(zz))
                    bp_max_indices, bp_min_indices = get_art_peaks(zz)

                    # use these indices to create mask of positions to use
                    # i.e. 1 if we want to use this position in the extra loss
                    # function, 0 otherwise
                    bp_loss_mask = np.zeros(zz.shape[0], dtype=np.float64)
                    bp_indices = np.array(bp_max_indices + bp_min_indices, dtype=np.int)
                    np.put(bp_loss_mask, bp_indices, 1.)
                    # add this to y so that we can use it in the loss function
                    zz_with_mask = np.stack((self.y_scaler.transform(zz.reshape(-1, 1))[:, 0], bp_loss_mask), axis=-1)

                    # merge waveform data with demographic data for input matrix
                    X_scaled = self.X_scaler.transform(X[idx:idx + self.window_len, 0:-1])
                    X_scaled = np.concatenate((X_scaled, X_demo), axis=1)
                    X_padded = np.pad(X_scaled, ((pad_size, pad_size), (0, 0)), 'edge')
                    batch_X.append(X_padded)
                    # either return waveform or scaler values as target to predict
                    if self.wave_or_scaler == "wave":
                        batch_y.append(zz_with_mask)
                    else:
                        median_sys_bp = self.y_scaler.transform(np.median(zz[bp_max_indices]).reshape(-1, 1))[0, 0]
                        median_dias_bp = self.y_scaler.transform(np.median(zz[bp_min_indices]).reshape(-1, 1))[0, 0]
                        mean_bp = self.y_scaler.transform(np.mean(zz).reshape(-1, 1))[0, 0]
                        batch_y.append([median_sys_bp, median_dias_bp, mean_bp])

        # Generate data
        #target = {"wave": np.array(batch_y), "bp": np.array(batch_y2)}
        target = np.array(batch_y)
        return np.array(batch_X), target
