import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import multiprocessing as mp
import random
import argparse

from tqdm import tqdm

sys.path.append("../../")
import src.utils as utils
import src.project_configs as project_configs


def get_nibp_art_values(file_name):
    save_dir = project_configs.stats_file_dir

    patient_id = os.path.basename(file_name).split("_")[0]
    try:
        df = pd.read_csv(file_name, sep=",", header=0, index_col=0, engine='c',
                         parse_dates=True, infer_datetime_format=True)
    except pd.errors.ParserError:
        print("C parser failed, falling back to Python parser...")
        df = pd.read_csv(file_name, sep=",", header=0, index_col=0, engine='python',
                         parse_dates=True, infer_datetime_format=True)
    except EOFError as e:
        print(e)
        print(file_name)
        return
    df.index = pd.to_datetime(df.index)

    # get rolling mean to smooth out signal using 10sec window
    df["art_rolling"] = df["art"].rolling(1000, center=True).mean()

    # get NIBP values and corresponding ART values
    if "NIBP Mean" in df.columns.values:
        nibp_mean_col_name = "NIBP Mean"
    elif "NIBP_mean" in df.columns.values:
        nibp_mean_col_name = "NIBP_mean"
    try:
        matching_data = df[(~df[nibp_mean_col_name].isna()) & (df["art"] > 10)]
    except KeyError as e:
        print(e)
        print("Guilty file:", file_name)
        return

    # save patient-specific distribution
    patient_save_file = os.path.join(save_dir, "{}_nibp_art_data.csv.gz".format(patient_id))
    # if we already have data for this patient, load old data and append
    if os.path.exists(patient_save_file):
        patient_df = pd.read_csv(patient_save_file, sep=",",
                                 header=0, index_col=0, engine='c',
                                 parse_dates=True, infer_datetime_format=True)
        matching_data = matching_data.append(patient_df)
    # save to file for later
    matching_data.to_csv(patient_save_file,
                         sep=",",
                         header=True,
                         index=True)


def create_nibp_art_files():
    # configs
    shuffle = False
    multithread = False

    print("Loading data from: {}".format(project_configs.preprocessed_data_dir))
    files = np.sort(np.unique(glob.glob(os.path.join(project_configs.preprocessed_data_dir, "*.csv.gz"))))

    # for running on hoffman
    try:
        task_id = int(os.environ["SGE_TASK_ID"])
        print("Task ID:", task_id)
        files = [files[task_id]]
        shuffle = False
    except KeyError:
        pass

    if shuffle:
        # shuffle the patient ID list
        random.shuffle(files)

    print(files)

    if multithread:
        #pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(4)
        print("using {} CPUs".format(mp.cpu_count()))
        results = pool.map(get_nibp_art_values, [ff for ff in files])
        pool.close()
    else:
        # iterate through all patients
        for ff in tqdm(files):
            print(ff)
            get_nibp_art_values(ff)


def plot_nibp_bland_altman():
    save_dir = project_configs.stats_file_dir
    preprocessed_files = np.unique(glob.glob(os.path.join(project_configs.stats_file_dir, "*_nibp_art_data.csv.gz")))
    print(preprocessed_files)

    art_mean_bp = np.array([])
    nibp_mean_bp = np.array([])

    for f in tqdm(preprocessed_files):
        patient_id = os.path.basename(f).split("-")[0]
        df = pd.read_csv(f, sep=",", header=0, index_col=0, engine='c', parse_dates=True, infer_datetime_format=True)
        df.index = pd.to_datetime(df.index)

        # keep results for bland-altman
        if "NIBP Mean" in df.columns.values:
            nibp_mean_col_name = "NIBP Mean"
        elif "NIBP_mean" in df.columns.values:
            nibp_mean_col_name = "NIBP_mean"
        art_mean_bp = np.append(art_mean_bp, df["art_rolling"].values)
        nibp_mean_bp = np.append(nibp_mean_bp, df[nibp_mean_col_name].values)

    # get differences
    diffs = art_mean_bp - nibp_mean_bp
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    # plot bland-altman and save to file
    plot_ylim = [-60, 60]
    plot_xlim = [0, 200]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    utils.bland_altman_plot(art_mean_bp, nibp_mean_bp, ax=ax)
    ax.set_title("Mean NIBP vs Mean ABP Bland-Altman Plot: {} +/- {}".format(
        np.round(mean_diff, 1), np.round(std_diff, 1)))
    ax.set_ylim(plot_ylim)
    ax.set_xlim(plot_xlim)
    plt.savefig(os.path.join(save_dir, "nibp_bland_altman_plot.png"))

    # save values to file
    pd.DataFrame({"art_mean_bp": art_mean_bp,
                  "nibp_mean_bp": nibp_mean_bp}).to_csv(os.path.join(save_dir, "nibp_bland_altman_values.csv.gz"),
                                                        sep=",",
                                                        header=True,
                                                        index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for creating NIBP vs ABP files for each patient. Note: "
                                                 "make sure that you delete all previous files since this script "
                                                 "appends to old files!")
    parser.add_argument('--plot', help='Create the NIBP vs ABP Bland-Altman figure', action='store_true', default=False)
    parser.add_argument('--clean', help='Remove old *_nibp_art_data.csv.gz files', action='store_true', default=False)
    parser.add_argument('--run', help='Create the NIBP vs ABP files', action='store_true', default=False)
    args = parser.parse_args()

    if args.plot:
        plot_nibp_bland_altman()
    elif args.clean:
        files = glob.glob(os.path.join(project_configs.stats_file_dir, "*_nibp_art_data.csv.gz"))
        print("removing files...")
        for f in files:
            os.remove(f)
    elif args.run:
        # NOTE: if running this part of the script, make sure that you delete all previous files
        # since this script appends to old files!
        create_nibp_art_files()


