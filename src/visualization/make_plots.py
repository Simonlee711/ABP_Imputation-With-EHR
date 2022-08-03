import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from tqdm import tqdm
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

sys.path.append("../")
sys.path.append("../../")
import src.project_configs as project_configs
import src.utils as utils


def bland_altman_per_patient(bland_altman_sys, bland_altman_dias,
                             sys_axis_lim=[50, 200], dias_axis_lim=[0, 150],
                             y_label="Invasive - Predicted Arterial Pressure [mmHg]",
                             x_label="(Invasive + Predicted Arterial Pressure)/2 [mmHg]",
                             title_string="Bland-Altman {} ABP - Test: {} +/- {}",
                             plot_file_name="joint_bland_altman.png",
                             save_dir="./"):
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

    population_rmse_sys = [np.sqrt(np.mean(np.square(m[1]))) for m in bland_altman_sys.values()]

    dias_diffs = [i for sublist in [m[1] for m in bland_altman_dias.values()] for i in sublist]
    mean_dias_diff_all = np.mean(dias_diffs)
    std_dias_diff_all = np.std(dias_diffs)

    population_rmse_dias = [np.sqrt(np.mean(np.square(m[1]))) for m in bland_altman_dias.values()]

    plot_lim = [-60, 60]
    line_limits = [1, 2]
    dashes = [[20, 5], [10, 2]]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].set_ylim(plot_lim)
    ax[0].set_xlim(sys_axis_lim)
    ax[0].set_yticks(np.arange(plot_lim[0], plot_lim[1] + 1, 15))
    ax[0].set_xticks(np.arange(sys_axis_lim[0], sys_axis_lim[1] + 1, 25))
    ax[0].tick_params(labelsize=project_configs.axis_tick_label_size)
    ax[0].errorbar(x_sys, y_sys, xerr=x_error_sys, yerr=y_error_sys, fmt='o', markeredgecolor='black',
                   ecolor='g', capthick=2, capsize=2)
    ax[0].set_ylabel(y_label, fontsize=project_configs.axis_label_font_size)
    ax[0].set_xlabel(x_label, fontsize=project_configs.axis_label_font_size)
    # add number of points to plot
    ax[0].legend(["N={}".format(len(bland_altman_sys))], loc='upper left')
    # add title
    ax[0].set_title(title_string.format("Systolic",
                                        np.round(mean_sys_diff_all, 1),
                                        np.round(std_sys_diff_all, 1)), fontsize=project_configs.title_font_size)
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
    ax[1].tick_params(labelsize=project_configs.axis_tick_label_size)
    ax[1].errorbar(x_dias, y_dias, xerr=x_error_dias, yerr=y_error_dias, fmt='o', markeredgecolor='black',
                   ecolor='g', capthick=2, capsize=2)
    ax[1].set_ylabel(y_label, fontsize=project_configs.axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=project_configs.axis_label_font_size)
    # add number of points to plot
    ax[1].legend(["N={}".format(len(bland_altman_dias))], loc='upper left')
    # add title
    ax[1].set_title(title_string.format("Diastolic",
                                        np.round(mean_dias_diff_all, 1),
                                        np.round(std_dias_diff_all, 1)), fontsize=project_configs.title_font_size)
    ax[1].axhline(mean_dias_diff_all, linestyle='-', c='black')
    for sd_limit, dash_style in zip(line_limits, dashes):
        limit_of_agreement = sd_limit * std_dias_diff_all
        lower = mean_dias_diff_all - limit_of_agreement
        upper = mean_dias_diff_all + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax[1].axhline(lim, linestyle='--', dashes=dash_style, c='black')
    plt.savefig(os.path.join(save_dir, plot_file_name))
    plt.show()
    plt.close()
    return ax


def error_vs_time_from_cuff(grouped_df, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    grouped_df["mean_rmse"].plot(ax=ax)
    ax.fill_between(np.arange(len(grouped_df["std_rmse"].values)),
                    (grouped_df["mean_rmse"] - grouped_df["std_rmse"]).values,
                    (grouped_df["mean_rmse"] + grouped_df["std_rmse"]).values,
                    alpha=0.2)

    # plt.xticks(rotation=45)
    # plt.locator_params(axis='x', nbins=20)
    ax.tick_params(labelsize=project_configs.axis_tick_label_size)
    plt.xlabel("Time from most recent NIBP measurement (seconds)", fontsize=project_configs.axis_label_font_size)
    plt.ylabel("RMSE", fontsize=project_configs.axis_label_font_size)
    plt.ylim([0, 80])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_vs_time_from_cuff.png"))
    plt.show()


def plot_long_waveform(patient_df, save_dir="./", show=False):
    fig, ax = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(16, 8))
    ax[0].plot(patient_df["y_true"], c='green', label="True ABP")

    ax[1].plot(patient_df["y_pred"], c='red', label="Predicted ABP")

    for a in ax:
        a.set_ylim((0, 200))
        a.tick_params(axis='both', which='major', labelsize=project_configs.axis_tick_label_size)
        a.legend(loc="upper right")

    ax[0].set_ylabel("Blood Pressure (mmHg)", fontsize=project_configs.axis_label_font_size)
    ax[1].set_ylabel("Blood Pressure (mmHg)", fontsize=project_configs.axis_label_font_size)

    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    labels = ax[1].get_xticks().tolist()
    labels = [int(x / project_configs.sample_freq) for x in labels]
    ax[1].set_xticklabels(labels, fontsize=project_configs.axis_tick_label_size)

    ax[1].set_xlabel("Time (s)", fontsize=project_configs.axis_label_font_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "{}_long_waveform.png".format(patient_df["patient_ID"].iloc[0])))
    if show:
        plt.show()
    plt.close()


def plot_long_waveforms(pred_dir, save_dir="./", show=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get all prediction files
    files = sorted(glob.glob(os.path.join(pred_dir, "*.csv.gz")))

    # get list of unique patient IDs
    patient_list = np.unique([os.path.basename(s).split("_")[0] for s in files])

    for p in tqdm(patient_list):
        if show:
            print(p)
        # get all files for patient p
        patient_files = glob.glob(os.path.join(pred_dir, "{}*.csv.gz".format(p)))
        # sort files by time
        patient_files_map = {int(os.path.basename(x).split("_")[1]): x for x in patient_files}
        patient_files = [x[1] for x in sorted(patient_files_map.items())]

        patient_dfs = []
        for f in patient_files:
            df = pd.read_csv(f)
            patient_dfs.append(df)

        patient_df = pd.concat(patient_dfs, ignore_index=True)
        patient_df.reset_index(inplace=True)
        patient_df.sort_values(by=["window_number", "window_count", "index"], inplace=True)

        plot_long_waveform(patient_df, save_dir=save_dir, show=show)


def plot_joint_error_from_cuff(pred_dir, dirs_to_consider, save_file="joint_rmse_vs_time_from_cuff.png"):
    """
    Plot for comparing error vs. time from cuff for multiple models jointly
    :param str pred_dir: path to directory with predictions
    :param list dirs_to_consider: list of directories in pred_dir to consider.
    Should be in order [Sideris et al., PPG scaling, V-net]
    :param str save_file: file name for saving plot
    :return: None
    """
    cuff_time_error = []
    for d in dirs_to_consider:
        f = os.path.join(pred_dir, d, "error_vs_time_from_cuff.csv")
        print(f)
        tdf = pd.read_csv(f, sep=",", header=0, index_col=0)
        tdf.columns = ["mean_rmse", "std_rmse"]
        tdf["data"] = d
        cuff_time_error.append(tdf)

    # needs to be in order: [sideris et al, ppg scaling, v-net]
    legend_names = ["Sideris et al.",
                    "PPG Scaling",
                    "V-Net"
                   ]

    dataset_name_mapping = {}
    for i in range(len(dirs_to_consider)):
        dataset_name_mapping[dirs_to_consider[i]] = legend_names[i]

    for i in cuff_time_error:
        i["data"] = i["data"].apply(lambda x: dataset_name_mapping[x])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in cuff_time_error:
        ax.plot(i["mean_rmse"], label=i["data"].values[0])
        ax.fill_between(np.arange(len(i["std_rmse"].values)),
                     (i["mean_rmse"] - i["std_rmse"]).values,
                     (i["mean_rmse"] + i["std_rmse"]).values,
                    alpha=0.3)
    plt.xlabel("Time from most recent NIBP measurement (seconds)")

    plt.xticks(np.arange(0, 60, step=6), rotation=45)
    plt.ylabel("RMSE")
    plt.ylim([0, 50])
    plt.legend(dataset_name_mapping.values())
    plt.tight_layout()

    plt.savefig(os.path.join("../reports/figures", save_file))
    plt.show()
    plt.close()


def main():
    predictions_dir = os.path.join(os.environ["HOME"], "code/github/ABP_pred/reports/predictions/2020-03-07_21.25.11_vnet_4s_no_noise_predictions/")

    save_dir = predictions_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading predictions from {}".format(predictions_dir))
    print("Saving results to {}".format(save_dir))

    # create Bland-Altman plots
    patient_bland_altman_sys = pickle.load(open(os.path.join(predictions_dir, "patient_bland_altman_sys.pkl"), "rb"))
    patient_bland_altman_dias = pickle.load(open(os.path.join(predictions_dir, "patient_bland_altman_dias.pkl"), "rb"))

    bland_altman_per_patient(patient_bland_altman_sys, patient_bland_altman_dias, save_dir=save_dir)

    # create error vs. time from cuff plots
    grouped_df = pd.read_csv(os.path.join(predictions_dir, "error_vs_time_from_cuff.csv"), header=0, index_col=0)
    error_vs_time_from_cuff(grouped_df, save_dir)

    # create error vs. BP variance plots
    patient_bp_var_error_sys = pickle.load(open(os.path.join(predictions_dir, "patient_bp_var_error_sys.pkl"), "rb"))
    patient_bp_var_error_dias = pickle.load(open(os.path.join(predictions_dir, "patient_bp_var_error_dias.pkl"), "rb"))
    bland_altman_per_patient(patient_bp_var_error_sys, patient_bp_var_error_dias,
                             sys_axis_lim=[-40, 40], dias_axis_lim=[-40, 40],
                             y_label="RMSE",
                             x_label="Variance of ABP",
                             title_string=" Error vs. Variance: {} Blood Pressure",
                             plot_file_name="error_vs_variance.png",
                             save_dir=save_dir)

    # create long waveform plots
    pred_dir = "/Volumes/Sammy/mimic_v9_4s/2020-03-07_21.25.11_vnet_4s_no_noise_predictions"
    long_waveform_save_dir = "../../reports/figures/long_waveforms"
    plot_long_waveforms(pred_dir, save_dir=long_waveform_save_dir, show=False)

    # create joint error vs. time from cuff plots
    dirs_to_consider = ["model_predictions_sideris_4s",
                        "ppg_baseline_4s_no_noise_test_patients",
                        "2020-03-07_21.25.11_vnet_4s_no_noise_predictions"
                        ]
    plot_joint_error_from_cuff(pred_dir="../reports/figures/predictions",
                               dirs_to_consider=dirs_to_consider,
                               save_file="4s_joint_rmse_vs_time_from_cuff.png")


if __name__ == "__main__":
    main()
