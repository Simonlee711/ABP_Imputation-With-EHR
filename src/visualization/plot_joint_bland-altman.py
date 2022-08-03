import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm

sys.path.append("../../")
import src.utils as utils
import src.project_configs as project_configs
import src.models.abp_model as abp_model

# file name configurations
model_weights = "../models/2019-12-14_08:02:50/weights.10.hdf5"
X_scaler_pickle = "../../models/train_X_scaler.pkl"
y_scaler_pickle = "../../models/train_y_scaler.pkl"
# number of consecutive windows to plot
num_batches_bland_altman = 100
# for plotting individual window input and output waveforms
plot_windows = True
train_or_val = "val"
save_dir = "bland-altman"
wave_or_scaler = "wave"


def bland_altman_vals(m1, m2):
    means = np.mean([m1, m2], axis=0)
    differences = np.array(m1) - np.array(m2)
    return means, differences


if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for the plots
    if train_or_val == "train":
        print("=" * 40)
        print("TRAINING SET")
        print("=" * 40)
        sys_title_string = "Bland-Altman - Systolic - Training: {} +/- {}"
        dias_title_string = "Bland-Altman - Diastolic - Training: {} +/- {}"
        fig_file_string = "epoch_{}_bland_altman_train_{}.png"
    elif train_or_val == "val":
        print("=" * 40)
        print("VALIDATION SET")
        print("=" * 40)
        sys_title_string = "Bland-Altman - Systolic - Validation: {} +/- {}"
        dias_title_string = "Bland-Altman - Diastolic - Validation: {} +/- {}"
        fig_file_string = "epoch_{}_bland_altman_val_{}.png"
    else:
        print("ERROR: train_or_val should be set to either 'train' or 'val'")

    # randomly select record from list of records
    record_list = glob.glob(os.path.join(project_configs.test_dir, "*.npy"))
    # create data frame
    record_df = pd.DataFrame(record_list, columns=["file_path"])
    # get list of patient IDs
    record_df["patient_ID"] = record_df["file_path"].apply(lambda x: os.path.basename(x).split("-")[0])

    # read in scaler objects
    X_train_scaler = pickle.load(open(X_scaler_pickle, "rb"))
    y_train_scaler = pickle.load(open(y_scaler_pickle, "rb"))
    # read in model and load weights
    model = abp_model.create_model_v2()
    model.load_weights(model_weights)

    bland_altman_sys = {}
    bland_altman_dias = {}
    # for each patient, iterate through all windows and predict waveform
    for p, window_files in tqdm(record_df.groupby("patient_ID")["file_path"]):

        y_true_sys_bp_all = []
        y_true_dias_bp_all = []
        y_pred_sys_bp_all = []
        y_pred_dias_bp_all = []

        for f in tqdm(window_files):
            rec = np.load(f, allow_pickle=True)
            num_windows = int(rec.shape[0] / project_configs.window_size)
            # if we do, use model to predict waveform for this continuous stretch
            input_batch = []
            output_batch = []
            pred_batch = []
            for i in range(num_windows):
                idx = i * project_configs.window_size

                input_window = np.pad(X_train_scaler.transform(rec[idx:idx + project_configs.window_size, 0:-1]),
                                      ((project_configs.padding_size, project_configs.padding_size), (0, 0)), 'edge')
                input_window = np.nan_to_num(input_window)
                # plt.plot(input_window)
                # plt.show()
                input_batch.append(input_window)
                # input_window = np.repeat(input_window[np.newaxis, :, :], project_configs.batch_size, axis=0)
                # print("input window dim: {}".format(input_window.shape))

                # get true BP waveform
                y_true = rec[idx:idx + project_configs.window_size, -1]
                output_batch.append(y_true)

                # get predicted BP waveform
                pred_abp = model.predict(np.array([input_window]), batch_size=1)[0]
                pred_batch.append(pred_abp)
            # # repeat array if we don't have enough samples for a batch
            # input_array = np.repeat(np.array(input_batch), np.ceil(project_configs.batch_size/num_windows), axis=0)
            # input_array = input_array[0:project_configs.batch_size, :, :]
            # get predicted BP waveform
            # pred_abp = model.predict(input_array, batch_size=project_configs.batch_size)

            # iterate through predictions in batch and
            for i in range(len(pred_batch)):
                y_true = output_batch[i]
                y_pred = pred_batch[i]
                # plt.plot(y_true)
                # plt.plot(y_pred)
                # plt.show()
                # get values of predicted wave at sys/dias BP indices
                if wave_or_scaler == "wave":
                    # scale BP back to normal units
                    y_true_scaled = y_true  # data does not need to be scaled
                    # y_true_scaled = y_train_scaler.inverse_transform(y_true)

                    # get predictions
                    y_pred_scaled = y_train_scaler.inverse_transform(y_pred)[:, 0]
                    # get indices of sys/dias BP
                    true_bp_max_indices, true_bp_min_indices = utils.get_art_peaks(y_true_scaled)
                    pred_bp_max_indices, pred_bp_min_indices = utils.get_art_peaks(y_pred_scaled)

                    # align bp indices in case of different number of peaks
                    true_bp_max_indices, pred_bp_max_indices = utils.align_lists(true_bp_max_indices, pred_bp_max_indices)
                    true_bp_min_indices, pred_bp_min_indices = utils.align_lists(true_bp_min_indices, pred_bp_min_indices)

                    # get values of blood pressure at peak indices
                    y_true_sys_bp_all = y_true_sys_bp_all + list(y_true_scaled[true_bp_max_indices])
                    y_true_dias_bp_all = y_true_dias_bp_all + list(y_true_scaled[true_bp_min_indices])

                    y_pred_sys_bp_all = y_pred_sys_bp_all + list(y_pred_scaled[pred_bp_max_indices])
                    y_pred_dias_bp_all = y_pred_dias_bp_all + list(y_pred_scaled[pred_bp_min_indices])
                    # BP prediction should be a tuple of scalers (systolic, diastolic, mean BP)
                elif wave_or_scaler == "scaler":
                    # scale BP back to normal units
                    y_true_scaled = y_true
                    # y_true_scaled = y_train_scaler.inverse_transform(y_true)
                    y_true_sys_bp_all = y_true_sys_bp_all + list([y_true_scaled[0]])
                    y_true_dias_bp_all = y_true_dias_bp_all + list([y_true_scaled[1]])
                    # get predictions
                    y_pred_scaled = y_train_scaler.inverse_transform(y_pred)
                    y_pred_sys_bp_all = y_pred_sys_bp_all + list([y_pred_scaled[0]])
                    y_pred_dias_bp_all = y_pred_dias_bp_all + list([y_pred_scaled[1]])

        # create two-part figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        # left window: systolic BP
        utils.bland_altman_plot(y_true_sys_bp_all, y_pred_sys_bp_all, ax=ax1)
        ax1.legend(["N={}".format(len(y_true_sys_bp_all))], loc='upper left')
        diffs = np.array(y_true_sys_bp_all) - np.array(y_pred_sys_bp_all)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, axis=0)
        ax1.title.set_text(sys_title_string.format(np.round(mean_diff, 1), np.round(std_diff, 1)))
        # right window: diastolic BP
        utils.bland_altman_plot(y_true_dias_bp_all, y_pred_dias_bp_all, ax=ax2)
        ax2.legend(["N={}".format(len(y_true_dias_bp_all))], loc='upper left')
        diffs = np.array(y_true_dias_bp_all) - np.array(y_pred_dias_bp_all)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, axis=0)
        ax2.title.set_text(dias_title_string.format(np.round(mean_diff, 1), np.round(std_diff, 1)))
        # save figure to file
        plt.savefig(os.path.join(save_dir, fig_file_string.format(p, wave_or_scaler)))
        #plt.show()
        plt.close()
        # write data used for bland-altman plot to file
        sys_bp_vals = {"sys_true": y_true_sys_bp_all,
                       "sys_pred": y_pred_sys_bp_all}
        save_file_name = os.path.join(save_dir, "epoch_{}_sys_bland-altman.csv".format(p))
        pd.DataFrame.from_dict(sys_bp_vals, orient="columns").to_csv(save_file_name, header=True, index=False)
        dias_bp_vals = {"dias_true": y_true_dias_bp_all,
                        "dias_pred": y_pred_dias_bp_all}
        save_file_name = os.path.join(save_dir, "epoch_{}_dias_bland-altman.csv".format(p))
        pd.DataFrame.from_dict(dias_bp_vals, orient="columns").to_csv(save_file_name, header=True, index=False)
        # save data for plotting bland-altman for all patients
        bland_altman_sys[p] = bland_altman_vals(y_true_sys_bp_all, y_pred_sys_bp_all)
        bland_altman_dias[p] = bland_altman_vals(y_true_dias_bp_all, y_pred_dias_bp_all)

        # plot bland-altman for all patients, where x-value is mean of the average, y-value is mean of
        # differences (errors), x-error-bar is std. dev. of average, and y-error-bar is std. dev. of the
        # differences (errors
        x_sys = [np.mean(m[0]) for m in bland_altman_sys.values()]
        y_sys = [np.mean(m[1]) for m in bland_altman_sys.values()]
        x_error_sys = [np.std(m[0]) for m in bland_altman_sys.values()]
        y_error_sys = [np.std(m[1]) for m in bland_altman_sys.values()]

        x_dias = [np.mean(m[0]) for m in bland_altman_dias.values()]
        y_dias = [np.mean(m[1]) for m in bland_altman_dias.values()]
        x_error_dias = [np.std(m[0]) for m in bland_altman_dias.values()]
        y_error_dias = [np.std(m[1]) for m in bland_altman_dias.values()]

        # draw lines on plot for mean, SD, 2xSD
        sys_diffs = [i for sublist in [m[1] for m in bland_altman_sys.values()] for i in sublist]
        mean_sys_diff_all = np.mean(sys_diffs)
        std_sys_diff_all = np.std(sys_diffs)
        print("Mean diff Sys BP: {} (+/- {})".format(mean_sys_diff_all, std_sys_diff_all))
        print("Mean Absolute diff Sys BP: {} (+/- {})".format(np.mean(np.abs(sys_diffs)), np.std(np.abs(sys_diffs))))

        dias_diffs = [i for sublist in [m[1] for m in bland_altman_dias.values()] for i in sublist]
        mean_dias_diff_all = np.mean(dias_diffs)
        std_dias_diff_all = np.std(dias_diffs)
        print("Mean diff Dias BP: {} (+/- {})".format(mean_dias_diff_all, std_dias_diff_all))
        print("Mean Absolute diff Dias BP: {} (+/- {})".format(np.mean(np.abs(dias_diffs)), np.std(np.abs(dias_diffs))))

        plot_lim = [-60, 60]
        sys_axis_lim = [50, 200]
        dias_axis_lim = [0, 150]
        axis_label_font_size = 14
        title_font_size = 16
        title_string = "Bland-Altman {} ABP - Validation: {} +/- {}"
        line_limits = [1, 2]
        dashes = [[20, 5], [10, 2]]


        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].set_ylim(plot_lim)
        ax[0].set_xlim(sys_axis_lim)
        ax[0].set_yticks(np.arange(plot_lim[0], plot_lim[1] + 1, 15))
        ax[0].set_xticks(np.arange(sys_axis_lim[0], sys_axis_lim[1] + 1, 25))
        ax[0].tick_params(labelsize=13)
        ax[0].errorbar(x_sys, y_sys, xerr=x_error_sys, yerr=y_error_sys, fmt='o', ecolor='g', capthick=2, capsize=2)
        ax[0].set_ylabel("Invasive - Predicted Arterial Pressure [mmHg]", fontsize=axis_label_font_size)
        ax[0].set_xlabel("(Invasive + Predicted Arterial Pressure)/2 [mmHg]", fontsize=axis_label_font_size)
        # add number of points to plot
        ax[0].legend(["N={}".format(len(bland_altman_sys))], loc='upper left')
        # add title
        ax[0].set_title(title_string.format("Systolic",
                                            np.round(mean_sys_diff_all, 1),
                                            np.round(std_sys_diff_all, 1)), fontsize=title_font_size)
        # add SD lines
        ax[0].axhline(mean_sys_diff_all, linestyle='-', c='black')
        for sd_limit, dash_style in zip(line_limits, dashes):
            limit_of_agreement = sd_limit * std_sys_diff_all
            lower = mean_sys_diff_all - limit_of_agreement
            upper = mean_sys_diff_all + limit_of_agreement
            for j, lim in enumerate([lower, upper]):
                ax[0].axhline(lim, linestyle='--', dashes=dash_style, c='black')


        ax[1].set_ylim(plot_lim)
        ax[1].set_xlim(dias_axis_lim)
        ax[1].set_yticks(np.arange(plot_lim[0], plot_lim[1] + 1, 15))
        ax[1].set_xticks(np.arange(dias_axis_lim[0], dias_axis_lim[1] + 1, 25))
        ax[1].tick_params(labelsize=13)
        ax[1].errorbar(x_dias, y_dias, xerr=x_error_dias, yerr=y_error_dias, fmt='o', ecolor='g', capthick=2, capsize=2)
        ax[1].set_ylabel("Invasive - Predicted Arterial Pressure [mmHg]", fontsize=axis_label_font_size)
        ax[1].set_xlabel("(Invasive + Predicted Arterial Pressure)/2 [mmHg]", fontsize=axis_label_font_size)
        # add number of points to plot
        ax[1].legend(["N={}".format(len(bland_altman_dias))], loc='upper left')
        # add title
        ax[1].set_title(title_string.format("Diastolic",
                                            np.round(mean_dias_diff_all, 1),
                                            np.round(std_dias_diff_all, 1)), fontsize=title_font_size)
        ax[1].axhline(mean_dias_diff_all, linestyle='-', c='black')
        for sd_limit, dash_style in zip(line_limits, dashes):
            limit_of_agreement = sd_limit * std_dias_diff_all
            lower = mean_dias_diff_all - limit_of_agreement
            upper = mean_dias_diff_all + limit_of_agreement
            for j, lim in enumerate([lower, upper]):
                ax[1].axhline(lim, linestyle='--', dashes=dash_style, c='black')
        plt.savefig("joint_bland_altman.png")
        plt.show()
