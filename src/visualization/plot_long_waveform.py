import os
import glob
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

import matplotlib as mpl
import platform
if platform.system() == 'Darwin':
    mpl.use('MacOSX')

from sklearn.preprocessing import RobustScaler

sys.path.append("../../")
import src.features.build_features as build_features
import src.project_configs as project_configs
import src.models.abp_model as abp_model
import src.utils as utils

# file name configurations
model_weights = "../models/2020-03-07_21:25:11_vnet_4s_no_noise/weights.29.hdf5"
# number of consecutive windows to plot
num_windows_plot = int(10*60/(project_configs.window_size/project_configs.sample_freq))
#num_windows_plot = int(1*60/(project_configs.window_size/project_configs.sample_freq))
# for plotting individual window input and output waveforms
plot_windows = True
# directory to save files
save_dir = "long_waveforms"

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # randomly select record from list of records
    record_list = glob.glob(os.path.join(project_configs.preprocessed_data_dir, "*.csv.gz"))

    # get demographic data
    demo_df = utils.get_demo_df()
    valid_patient = False
    while not valid_patient:
        rand_rec = random.choice(record_list)
        # get demographic data for this patient
        try:
            patient_id = os.path.basename(rand_rec).split("-")[0]
            X_demo = np.repeat(demo_df.loc[patient_id], project_configs.window_size).values.reshape(
                project_configs.window_size, demo_df.shape[1])
            valid_patient = True
        except KeyError as e:
            print(e)

    print("Randomly selected {} for plotting".format(rand_rec))

    # read in scaler objects
    X_train_scaler, y_train_scaler = utils.load_scaler_objects(project_configs.X_scaler_pickle,
                                                               project_configs.y_scaler_pickle)
    # read in model and load weights
    model = abp_model.create_model_vnet()
    model.load_weights(model_weights)

    # randomly index into record
    #max_index = rec.shape[0] - (num_windows_plot * project_configs.window_size)

    # find a continuous stretch of valid windows
    chunk_size = project_configs.window_size * num_windows_plot
    chunk_counter = 0

    total_valid_data = pd.DataFrame()
    # read record in chunks
    for rec in pd.read_csv(os.path.join(project_configs.preprocessed_data_dir, rand_rec), sep=",",
                           header=0, index_col=0, chunksize=chunk_size):

        # trim useless signal and create additional features used in model
        rec = build_features.create_additional_features(rec)[0]

        # keep track of all windows in this chunk
        chunk_df = pd.DataFrame()

        for i in range(num_windows_plot):
            # if we do, use model to predict waveform for this continuous stretch
            idx = (i * project_configs.window_size)

            # correct signal lag
            corr_coeffs, offset = build_features.find_signal_shift(rec["sp02"][idx:idx + project_configs.shift_window_size],
                                                    rec["art"][idx:idx + project_configs.shift_window_size], max_shift=400)

            xx = rec["ekg"][idx:idx + project_configs.window_size]
            #yy = rec["sp02"].shift(offset)[idx:idx + project_configs.window_size]
            yy = rec.loc[:, rec.columns.str.startswith("sp02")].shift(offset)[i:i + project_configs.window_size]
            zz = rec["art"][idx:idx + project_configs.window_size]
            rec.loc[:, rec.columns.str.startswith("sp02")] = yy

            if xx.shape[0] == 0 or yy.shape[0] == 0 or zz.shape[0] == 0:
                print("xx shape: {} yy shape: {} zz shape: {}".format(xx.shape, yy.shape, zz.shape))
                continue

            window_is_valid, reason = build_features.is_valid_window(xx.values, yy["sp02"].values, zz.values)

            # additional check: make sure NIBP values are reasonable
            if window_is_valid:
                nibp_sys = rec[project_configs.nibp_column_names[0]][idx:idx + project_configs.window_size].values.mean()
                nibp_dias = rec[project_configs.nibp_column_names[1]][idx:idx + project_configs.window_size].values.mean()
                nibp_mean = rec[project_configs.nibp_column_names[2]][idx:idx + project_configs.window_size].values.mean()

                nibp_window_is_valid, nibp_reason = build_features.is_valid_nibp(nibp_sys, nibp_dias, nibp_mean)
                if not nibp_window_is_valid:
                    window_is_valid = False
                    reason = reason + "Invalid NIBP: " + nibp_reason

            # if window is valid, we color the predicted signal red
            if window_is_valid:
                wave_color = "red"
                total_valid_data = total_valid_data.append(rec.iloc[idx:idx + project_configs.window_size, 0:2])
            # if the window is invalid, we color the predicted signal blue
            else:
                wave_color = "blue"
                print("INVALID WINDOW:", reason)
                plt.plot(yy)
                # plt.show()
                plt.close()

            # draw random noise value and add to pseudo-NIBP to simulate cuff error
            if project_configs.add_nibp_noise:
                noise_val = np.random.normal(loc=11.8, scale=28.9, size=1)
                rec.iloc[idx:idx + project_configs.window_size, -4:-1] += float(noise_val)

            # generate predicted waveform for this window
            # scale waveforms
            if window_is_valid:
                scaler = RobustScaler()
                scaler.fit(total_valid_data.iloc[:, 0:2])
                rec.iloc[idx:idx + project_configs.window_size, 0:2] = scaler.transform(
                    rec.iloc[idx:idx + project_configs.window_size, 0:2])

            X_scaled = X_train_scaler.transform(rec.iloc[idx:idx + project_configs.window_size, 0:-1])
            X_scaled = np.concatenate((X_scaled, X_demo), axis=1)
            input_window = np.pad(X_scaled, ((project_configs.padding_size, project_configs.padding_size),
                                             (0, 0)), 'edge')

            input_window = np.nan_to_num(input_window)
            print("input window dim: {}".format(input_window.shape))
            # plt.plot(input_window)
            # plt.show()
            pred_abp = model.predict(np.array([input_window]), batch_size=1)
            pred_abp_waveform = pred_abp[0]

            # create a data frame with the true and predicted waveform
            abp_df = pd.DataFrame(rec.iloc[idx:idx + project_configs.window_size, project_configs.abp_col].values,
                                  columns=["True ABP"])
            print("pred abp windows shape: {}".format(np.array(pred_abp_waveform).shape))
            abp_df["Pred ABP"] = y_train_scaler.inverse_transform(np.array(pred_abp_waveform))
            abp_df["color"] = wave_color
            # append this window to dataframe for all windows in chunk
            chunk_df = chunk_df.append(abp_df)

        # for this chunk, plot the true and predicted, coloring the predicted signal red in valid windows
        # and blue in invalid windows
        chunk_df.reset_index(inplace=True)
        chunk_df.to_csv(os.path.join(save_dir, "long_waveform_data_chunk_{}.csv".format(chunk_counter)),
                        sep=",", header=True, index=False)
        # abp_df.plot()
        print(chunk_df.head())
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16, 8))
        ax1.plot(chunk_df["True ABP"], c="green")
        ax1.legend(["True ABP"])
        chunk_df["Pred ABP"].plot(color=chunk_df["color"].values, ax=ax2)
        ax2.legend(["Pred ABP"])
        plt.savefig(os.path.join(save_dir, "long_waveform_chunk_{}.png".format(chunk_counter)))
        # plt.show()
        plt.close()

        chunk_counter += 1




