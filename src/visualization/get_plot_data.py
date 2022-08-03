import os
import pandas as pd
import numpy as np
import glob
import sys
from tqdm import tqdm
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

sys.path.append("../../")
import src.project_configs as project_configs
import src.utils as utils

results_string = "{:.2f} ({:.2f}, {:.2f}) ± {:.2f} ({:.2f}, {:.2f})"


def bland_altman_vals(y_true, y_pred):
    means = np.mean([y_true, y_pred], axis=0)
    differences = np.array(y_true) - np.array(y_pred)
    return means, differences


def wave_rmse(window):
    if window.shape[0] > 0:
        win = window[np.all(~window[["y_true", "y_pred"]].isna(), axis=1)]
        return np.sqrt(mean_squared_error(win["y_true"], win["y_pred"]))
    else:
        return np.nan


def wave_corr(window):
    if window.shape[0] > 0:
        win = window[np.all(~window[["y_true", "y_pred"]].isna(), axis=1)]
        return np.corrcoef(win["y_true"], win["y_pred"])[0, 1]
    else:
        return np.nan


def wave_mae(window):
    if window.shape[0] > 0:
        win = window[np.all(~window[["y_true", "y_pred"]].isna(), axis=1)]
        return mean_absolute_error(win["y_true"], win["y_pred"])
    else:
        return np.nan


def get_pred_data(predictions_dir):
    """
    Merge waveform predictions for each patient into single dataframe
    :param predictions_dir: path to directory containing prediction files
    :return: pandas DataFrame with concatenated predictions
    """
    pred_files = glob.glob(os.path.join(predictions_dir, "*.csv.gz"))
    print("Found {} files".format(len(pred_files)))

    results_dfs_list = []
    for f in tqdm(pred_files):
        try:
            df = pd.read_csv(f, sep=",", header=0)
            results_dfs_list.append(df)
        except:
            print("Count not find any data in {}".format(f))

    # concatenate per patient data into single dataframe
    results_df = pd.concat(results_dfs_list)
    del results_dfs_list
    return results_df


def get_beat_bp_vals(window):
    """
    Takes a window with true and predicted BP waveforms and returns pairs of
    predicted/true systolic/diastolic BP values
    :param window: pandas DataFrame with window of true (y_true) and predicted (y_pred) waveforms
    :return: (list, list, list, list) with true systolic, pred systolic, true diastolic, and pred diastolic BP values
    """
    # get indices of sys/dias BP
    true_bp_max_indices, true_bp_min_indices = utils.get_art_peaks(window["y_true"])
    pred_bp_max_indices, pred_bp_min_indices = utils.get_art_peaks(window["y_pred"])

    # align bp indices in case of different number of peaks
    true_bp_max_indices, pred_bp_max_indices = utils.align_lists(true_bp_max_indices, pred_bp_max_indices)
    true_bp_min_indices, pred_bp_min_indices = utils.align_lists(true_bp_min_indices, pred_bp_min_indices)

    # get values of blood pressure at peak indices
    y_true_sys_bp_all = window.iloc[true_bp_max_indices, :]["y_true"].values
    y_true_dias_bp_all = window.iloc[true_bp_min_indices, :]["y_true"].values

    y_pred_sys_bp_all = window.iloc[pred_bp_max_indices, :]["y_pred"].values
    y_pred_dias_bp_all = window.iloc[pred_bp_min_indices, :]["y_pred"].values

    return y_true_sys_bp_all, y_pred_sys_bp_all, y_true_dias_bp_all, y_pred_dias_bp_all


def wave_rmse_mean_std(window):
    """
    For calculating error distribution for each bin of times from most recent NIBP measurement
    :param window: pandas DataFrame with true and predicted waveforms for each patient
    :return: pandas Series with mean of RMSE and std of RMSE over all patients
    """
    if window.shape[0] > 0:
        win = window[np.all(~window[["y_true", "y_pred"]].isna(), axis=1)]
        mean_rmse = win.groupby("patient_ID").apply(
            lambda x: np.sqrt(mean_squared_error(x["y_true"], x["y_pred"]))).mean()
        std_rmse = win.groupby("patient_ID").apply(
            lambda x: np.sqrt(mean_squared_error(x["y_true"], x["y_pred"]))).std()
        return pd.Series({'mean_rmse': mean_rmse, 'std_rmse': std_rmse})
    else:
        return np.nan


def var_diff_vals(y_true, y_pred):
    """
    For calculating RMSE as a function of variance
    :param y_true: true blood pressure values
    :param y_pred: predicted blood pressure values
    :return: variance of true values, RMSE of true and predicted values
    """
    var = np.var(y_true)
    differences = np.sqrt(mean_squared_error(y_true, y_pred))
    return var, differences


def get_waveform_stats(results_df, save_dir):
    """
    Get waveform-level stats for each patient
    :param results_df: pandas DataFrame containing true and predicted waveforms for all patients
    :param save_dir: directory to save results
    :return: None
    """
    # log to file
    with open(os.path.join(save_dir, "wave_error_stats.txt"), "w") as stats_f:
        metrics = {"RMSE": wave_rmse, "Corr": wave_corr, "MAE": wave_mae}
        for k, v in metrics.items():
            # calculate root mean square error for each patient
            per_patient_res = results_df.groupby("patient_ID").apply(v)
            mean_bootstrap_res = bs.bootstrap(np.array(per_patient_res.values), stat_func=bs_stats.mean,
                                              iteration_batch_size=1)
            std_bootstrap_res = bs.bootstrap(np.array(per_patient_res.values), stat_func=bs_stats.std,
                                             iteration_batch_size=1)
            results_string_formatted = results_string.format(mean_bootstrap_res.value,
                                                             mean_bootstrap_res.lower_bound,
                                                             mean_bootstrap_res.upper_bound,
                                                             std_bootstrap_res.value,
                                                             std_bootstrap_res.lower_bound,
                                                             std_bootstrap_res.upper_bound)
            metric_results = "{}: {}\n".format(k, results_string_formatted)
            print(metric_results)
            stats_f.write(metric_results)
            per_patient_res.to_csv(os.path.join(save_dir, "per_patient_waveform_{}.csv".format(k)),
                                   header=True, index=True)


def get_bland_altman_data(results_df, save_dir, log_file_name="beat_error_stats.txt"):
    """
    Calculate bland-altman scores for each patient and save results as pickle object
    :param results_df: pandas DataFrame containing true and predicted waveforms for all patients
    :param save_dir: directory to save results
    :param log_file_name: name of file for saving metrics
    :return: None
    """
    unique_patients = results_df["patient_ID"].unique()
    patient_bland_altman_sys = {p: [[], []] for p in unique_patients}
    patient_bland_altman_dias = {p: [[], []] for p in unique_patients}

    for idx, val in tqdm(results_df.groupby(["patient_ID", "window_number"])):
        # get true and predicted systolic, diastolic values for each window for each patient
        sys_true, sys_pred, dias_true, dias_pred = get_beat_bp_vals(val)
        # get BP differences and means
        bland_altman_sys = bland_altman_vals(sys_true, sys_pred)
        bland_altman_dias = bland_altman_vals(dias_true, dias_pred)

        patient_bland_altman_sys[idx[0]][0].append(bland_altman_sys[0])
        patient_bland_altman_sys[idx[0]][1].append(bland_altman_sys[1])
        patient_bland_altman_dias[idx[0]][0].append(bland_altman_dias[0])
        patient_bland_altman_dias[idx[0]][1].append(bland_altman_dias[1])

    for p, v in patient_bland_altman_sys.items():
        v[0] = np.concatenate(v[0])
        v[1] = np.concatenate(v[1])

    for p, v in patient_bland_altman_dias.items():
        v[0] = np.concatenate(v[0])
        v[1] = np.concatenate(v[1])

    # dump data to file
    pickle.dump(patient_bland_altman_sys, open(os.path.join(save_dir, "patient_bland_altman_sys.pkl"), "wb"))
    pickle.dump(patient_bland_altman_dias, open(os.path.join(save_dir, "patient_bland_altman_dias.pkl"), "wb"))

    # calculate metrics
    sys_diffs = [i for sublist in [m[1] for m in patient_bland_altman_sys.values()] for i in sublist]
    #population_rmse_sys = [np.sqrt(np.mean(np.square(m[1]))) for m in patient_bland_altman_sys.values()]

    dias_diffs = [i for sublist in [m[1] for m in patient_bland_altman_dias.values()] for i in sublist]
    #population_rmse_dias = [np.sqrt(np.mean(np.square(m[1]))) for m in patient_bland_altman_dias.values()]

    vals = {"Mean diff Sys BP: {}\n": sys_diffs,
            #"Mean Abs diff Sys BP: {}\n": np.abs(sys_diffs),
            #"RMSE diff Sys BP: {}\n": population_rmse_sys,
            "Mean diff Dias BP: {}\n": dias_diffs,
            #"Mean Absolute diff Dias BP: {}\n": np.abs(dias_diffs),
            #"RMSE diff Dias BP: {}\n": population_rmse_dias}
            }
    with open(os.path.join(save_dir, log_file_name), "w") as out_f:
        for k, v in vals.items():
            mean_bootstrap_res = bs.bootstrap(np.array(v), stat_func=bs_stats.mean, iteration_batch_size=1)
            std_bootstrap_res = bs.bootstrap(np.array(v), stat_func=bs_stats.std, iteration_batch_size=1)
            txt = k.format(results_string.format(mean_bootstrap_res.value,
                                                 mean_bootstrap_res.lower_bound,
                                                 mean_bootstrap_res.upper_bound,
                                                 std_bootstrap_res.value,
                                                 std_bootstrap_res.lower_bound,
                                                 std_bootstrap_res.upper_bound
                                                 )
                           )

            print(txt)
            out_f.write(txt)


def get_error_vs_time_from_cuff(results_df, save_dir):
    """
    Calculate error as a function of time since last NIBP (cuff) measurement and save results
    :param results_df: pandas DataFrame containing true and predicted waveforms for all patients
    :param save_dir: directory to save results
    :return:
    """
    # max_time_from_nibp is maximum number of mins from cuff to plot
    max_time_from_nibp = 100 * 60 * 10
    # bin_width is number of seconds per bin
    bin_width = 10
    bins = range(0, int(max_time_from_nibp / project_configs.sample_freq), bin_width)

    # only get samples that are within reasonable time range
    results_df_filtered = results_df[results_df["prox"] < max_time_from_nibp]
    results_df_filtered["prox"] = results_df_filtered["prox"] / project_configs.sample_freq

    # group data by bin
    grouped_df = results_df_filtered.groupby(pd.cut(results_df_filtered["prox"], bins=bins)).apply(wave_rmse_mean_std)

    # write data to file
    grouped_df.index = grouped_df.index.astype(str)
    grouped_df.to_csv(os.path.join(save_dir, "error_vs_time_from_cuff.csv"), header=True, index=True)


def get_error_vs_variance(results_df, save_dir):
    """
    Get error as a function of variance and save data as pickle object
    :param results_df: pandas DataFrame containing true and predicted waveforms for all patients
    :param save_dir: directory to save results
    :return: None
    """
    unique_patients = results_df["patient_ID"].unique()
    patient_bp_var_error_sys = {p: [[], []] for p in unique_patients}
    patient_bp_var_error_dias = {p: [[], []] for p in unique_patients}

    for idx, val in results_df.groupby(["patient_ID", "window_number"]):
        sys_true, sys_pred, dias_true, dias_pred = get_beat_bp_vals(val)

        try:
            patient_bp_var_error_sys[idx[0]][0].append(var_diff_vals(sys_true, sys_pred)[0])
            patient_bp_var_error_sys[idx[0]][1].append(var_diff_vals(sys_true, sys_pred)[1])
            patient_bp_var_error_dias[idx[0]][0].append(var_diff_vals(dias_true, dias_pred)[0])
            patient_bp_var_error_dias[idx[0]][1].append(var_diff_vals(dias_true, dias_pred)[1])
        except ValueError:
            print("No valid data for {}".format(idx))

    # dump data to file
    pickle.dump(patient_bp_var_error_sys, open(os.path.join(save_dir, "patient_bp_var_error_sys.pkl"), "wb"))
    pickle.dump(patient_bp_var_error_dias, open(os.path.join(save_dir, "patient_bp_var_error_dias.pkl"), "wb"))


def main():
    # if len(sys.argv) > 1:
    if True:
        predictions_dir = sys.argv[1]
    else:
        predictions_dir = "/Volumes/Sammy/mimic_v9_4s/2020-03-07_21.25.11_vnet_4s_no_noise_predictions"

    if not os.path.exists(predictions_dir):
        raise OSError("{} does not exist!".format(predictions_dir))

    save_dir = os.path.join("../../reports/filtered_predictions/",
                            os.path.basename(predictions_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading predictions from {}".format(predictions_dir))
    print("Saving results to {}".format(save_dir))

    results_df = get_pred_data(predictions_dir)
    print("results_df.shape: ", results_df.shape)

    # get waveform-level stats for each patient
    print("Getting waveform stats...")
    get_waveform_stats(results_df, save_dir)

    # calculate bland-altman scores per patient
    print("Getting Bland-Altman score values...")
    get_bland_altman_data(results_df, save_dir)

    # calculate error as function of time from most recent NIBP measurement
    print("Getting error vs. time from cuff values...")
    get_error_vs_time_from_cuff(results_df, save_dir)

    # calculate error as function of BP variance in window
    print("Getting error vs. variance values...")
    #get_error_vs_variance(results_df, save_dir)

    print("Done.")


if __name__ == "__main__":
    main()
