import numpy as np
import pandas as pd
import os
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(os.path.join(os.environ["HOME"], "code/github/ABP_pred"))
import src.project_configs as project_configs
from src.utils import *
import src.utils as utils


add_noise = project_configs.add_nibp_noise
overwrite = False

train_or_val_or_test = "test"
if add_noise:
    save_dir = os.path.join(project_configs.project_dir,
                            "ppg_baseline_noise_" + os.path.basename(project_configs.test_dir))
else:
    save_dir = os.path.join(project_configs.project_dir,
                            "ppg_baseline_no_noise_" + os.path.basename(project_configs.test_dir))
wave_or_scaler = "wave"


if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # randomly select record from list of records
    if train_or_val_or_test == "train":
        record_list = glob.glob(os.path.join(project_configs.train_dir, "*.npy"))
    elif train_or_val_or_test == "val":
        record_list = glob.glob(os.path.join(project_configs.val_dir, "*.npy"))
    elif train_or_val_or_test == "test":
        record_list = glob.glob(os.path.join(project_configs.test_dir, "*.npy"))
    else:
        raise ValueError("ERROR: train_or_val must either be 'train' or 'val' or 'test'.")

    # create data frame
    record_df = pd.DataFrame(record_list, columns=["file_path"])
    # get list of patient IDs
    record_df["patient_ID"] = record_df["file_path"].apply(lambda x: utils.get_patient_from_file(x))
    record_df["win_num"] = record_df["file_path"].apply(lambda x: utils.get_window_from_file(x))

    # sort window files so that we process in chronological order
    record_df.sort_values(by=["patient_ID", "win_num"], inplace=True)

    error_count = 0
    total_num_windows = 0

    # for each patient, iterate through all windows and predict waveform
    for p, window_files in tqdm(record_df.groupby("patient_ID")["file_path"]):
        window_count = 0

        for f in tqdm(window_files[:min(len(window_files), 50)]):
        #for f in tqdm(window_files):
            rec = np.load(f, allow_pickle=True)
            num_windows = int(rec.shape[0] / project_configs.window_size)
            window_num = utils.get_window_from_file(f)
            # skip this file if it already exists and we don't want to overwrite
            if os.path.exists(os.path.join(save_dir, "{}_{}_predictions.csv.gz".format(p, window_num))) and not overwrite:
                continue
            result_df = pd.DataFrame()
            # if we do, use model to predict waveform for this continuous stretch
            for i in range(num_windows):
                window_count += 1
                idx = i * project_configs.window_size

                nibp_sys = rec[idx:idx + project_configs.window_size, project_configs.nibp_sys_col].mean()
                nibp_dias = rec[idx:idx + project_configs.window_size, project_configs.nibp_dias_col].mean()
                nibp_mean = rec[idx:idx + project_configs.window_size, project_configs.nibp_mean_col].mean()

                # Fix for build_features.py bug..
                if nibp_sys <= 0 or nibp_dias <= 0 or nibp_mean <= 0:
                    continue

                if add_noise:
                    noise_val = float(np.random.normal(loc=11.8, scale=28.9, size=1))
                    nibp_sys += noise_val
                    nibp_dias += noise_val

                # use MinMaxScaler to stretch PPG signal using NIBP sys, dias values
                scaler = MinMaxScaler(feature_range=(nibp_dias, nibp_sys))

                ecg = rec[idx:idx + project_configs.window_size, project_configs.ecg_col]
                ppg = rec[idx:idx + project_configs.window_size, project_configs.ppg_col]
                abp = rec[idx:idx + project_configs.window_size, project_configs.abp_col]
                try:
                    ppg_scaled = scaler.fit_transform(ppg.reshape(-1, 1))[:, 0]
                except ValueError as e:
                    print(e)
                    error_count += 1
                    continue

                # get proximity from NIBP measurement
                prox = rec[idx:idx + project_configs.window_size, project_configs.prox_col]

                y_true = abp
                y_pred_scaled = ppg_scaled

                # create dataframe with prediction results
                pred_df = pd.DataFrame.from_dict({"y_true": list(y_true),
                                                  "y_pred": list(y_pred_scaled),
                                                  "prox": list(prox),
                                                  "ecg": list(ecg),
                                                  "ppg": list(ppg)},
                                                 orient="columns")
                # add additional columns
                pred_df["nibp_sys"] = nibp_sys
                pred_df["nibp_dias"] = nibp_dias
                pred_df["patient_ID"] = p
                pred_df["window_number"] = window_count
                pred_df["date"] = "-".join((os.path.basename(f).split("_")[0]).split("-")[1:])
                # append to result dataframe
                result_df = result_df.append(pred_df)

            result_df.to_csv(os.path.join(save_dir, "{}_{}_predictions.csv.gz".format(p, window_num)), sep=",",
                             header=True, index=False)
