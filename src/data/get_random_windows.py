import sys
import os
import glob
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from tqdm import tqdm
import tensorflow as tf

sys.path.append("../../")
import src.project_configs as project_configs
import src.features.build_features as features


def get_ppg_qi(X, ppg_qi, ppg_qi_scaler, ppg_window_size=400):
    """Generate the minimum PPG quality index (QI) scale for a window of 
    PPG signal. A higher PPG QI value means the signal in the window
    is predicted to be of higher quality (better).

    Args:
        X ([type]): array containing raw PPG signal
        ppg_qi ([type]): keras PPG QI model
        ppg_qi_scaler ([type]): sklearn StandardScaler object fit on PPG QI training data
        ppg_window_size (int, optional): Number of PPG samples in PPG QI window. Defaults to 400.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
        
    if len(X.shape) < 2:
        X = np.expand_dims(X, -1)
        
    if len(X.shape) < 3:
        X = np.expand_dims(X, 0)
        
    X = (X - X.mean()) / X.std()
    
    # calculate number of PPG QI windows within input window 
    # for example, if we have 32s windows, we have 8 PPG QI predictions 
    # per window/row
    num_windows = int(X.shape[1] / ppg_window_size)

    ppg_window_preds = []
    for i in range(num_windows):
#         print(int(i*ppg_window_size), int(i*ppg_window_size + ppg_window_size))
        ppg_window_preds.append(ppg_qi.predict( np.expand_dims(X[:, int(i*ppg_window_size):int(i*ppg_window_size + ppg_window_size), 0], -1) )[:, 0])
#         ppg_window_preds.append(ppg_qi.predict( np.expand_dims(ppg_qi_scaler.transform(X[:, int(i*ppg_window_size):int(i*ppg_window_size + ppg_window_size), 0]), -1) )[:, 0])

    # stack each pred, and take min across windows (rows) to get worst case PPG QI in each window
    ppg_qi_preds = np.min(np.array(ppg_window_preds).flatten())
    
    return ppg_qi_preds


def main():
    parser = argparse.ArgumentParser(description="Get random windows from the preprocessed .csv files and save to .csv files")
    parser.add_argument('--data-dir', 
        help='Directory containing the preprocessed .csv.gz MIMIC waveform files', 
        required=True)
    parser.add_argument('--save-dir', 
        help='Directory to save the output window files', 
        required=True)
    parser.add_argument('--overwrite', 
        help='Flag to overwrite files; otherwise, skip records if file exists', 
        action='store_true')
    parser.add_argument('--shuffle', 
        help='Flag to shuffle the order of data files', 
        action='store_true')
    parser.add_argument('--window-size', 
        help='Number of seconds in window', 
        default=30, type=int)
    parser.add_argument('--max-iter-per-file', 
        help='Number of windows to try per file', 
        default=100000, type=int)
    args = parser.parse_args()

    window_size = int(args.window_size * project_configs.sample_freq)
    print("Using window size of {}s ({} samples @ {}Hz)".format(
        args.window_size, window_size, project_configs.sample_freq))
    max_num_iter = args.max_iter_per_file
    # get list of all files in data dir
    data_files = glob.glob(os.path.join(args.data_dir, "*.csv.gz"))
    if args.shuffle:
        print("Shuffling order of data files...")
        np.random.shuffle(data_files)
    print("Found {} data files".format(len(data_files)))

    # create directories for saving files if they don't exist
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # create directory to save the window data (.csv files)
    data_save_dir = os.path.join(save_dir, "data")
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir, exist_ok=True)
    # create directory to save the window plots (.png files)
    plot_save_dir = os.path.join(save_dir, "plots")
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    

    # configs for PPG QI
    ppg_qi_model_file = "../../../PPG_QI/weights/weights.ppgqi.hdf5"
    ppg_scaler_pkl_file = "../../../PPG_QI/weights/ppg_qi_scaler.pkl"
    threshold = 0.8108692169189453 # threshold determined on MIMIC validation data such that precision=0.95

    # load PPG QI model from file
    ppg_qi = tf.keras.models.load_model(ppg_qi_model_file)

    # load PPG QI scaler object
    ppg_qi_scaler = pickle.load(open(ppg_scaler_pkl_file, "rb"))

    # for each file, open and grab random windows until we have one with clean signal
    for f in tqdm(data_files):
        file_basename = os.path.basename(f).split("_merged")[0]
        data_f = os.path.join(data_save_dir, "{}_window_data.csv.gz".format(file_basename))
        plot_f = os.path.join(plot_save_dir, "{}_window_plot.png".format(file_basename))

        if os.path.exists(data_f) and not args.overwrite:
            print("{} exists and overwrite flag not used. Skipping...".format(data_f))
            continue

        df = pd.read_csv(f, index_col=0, nrows=100*60*60*48)
        print(df.head())

        signals = ["ECG", "PLETH", "ABP", "RESP"]
        num_iter = 0

        # keep testing windows until we find a valid one
        while num_iter < max_num_iter:
            random_idx = np.random.randint(low=0, high=df.shape[0]-window_size)

            window = df.iloc[random_idx:random_idx+window_size]
            
            # check that the signal is valid according to filtering rules
            window_is_valid, reason = features.is_valid_window(ekg=window["ECG"], 
                                                            spo2=window["PLETH"], 
                                                            art=window["ABP"])
            print(window_is_valid, reason)
            
            # if window passes rules, check PPG QI
            if window_is_valid:
            
                # get the minimum PPG QI value for the window
                ppg_window = window["PLETH"]
                ppg_qi_value = get_ppg_qi(ppg_window, ppg_qi, ppg_qi_scaler)
                print(random_idx, ppg_qi_value)

                # if PPG QI is better than cutoff threshold, save window
                if ppg_qi_value > threshold:

                    # save data
                    window.to_csv(data_f, header=True, index=True)

                    # save fig
                    fig, ax = plt.subplots(len(signals), 1, figsize=(16, 12))
                    for i, sig in enumerate(signals):
                        window[sig].plot(ax=ax[i], label=sig,)
                        ax[i].legend(loc="upper right")

                    plt.tight_layout()
                    plt.savefig(plot_f)

                    break
            num_iter += 1

    

if __name__ == "__main__":
    main()
