import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import random
from tqdm import tqdm

sys.path.append("../../")
import src.utils as utils
import src.project_configs as project_configs


# Configs
home_dir = os.environ["HOME"]
train_file_dir = project_configs.train_dir
val_file_dir = project_configs.val_dir
test_file_dir = project_configs.test_dir
wave_stats_dir = project_configs.stats_file_dir

# length of window in seconds
window_length = int(project_configs.window_size / project_configs.sample_freq)
# index of arterial blood pressure waveform in dataframe
aline_index = project_configs.abp_col
# patients need at least this many valid minutes of waveform data, otherwise they're removed
valid_minutes_threshold = 10


# stats we want to keep track of for all windows:
# mean/std of systolic, diastolic, mean arterial pressure
# min/max values?
def get_window_bp_vals(files, sample_freq=100):
    sys_bp_values = []
    dias_bp_values = []
    mean_bp_values = []

    for f in tqdm(files):
        arr = np.load(f)
        num_samples = (window_length*sample_freq)
        num_windows = int(arr.shape[0]/num_samples)
        for i in range(num_windows):
            window = arr[i*num_samples:(i+1)*num_samples, aline_index]
            sys_indices, dias_indices = utils.get_art_peaks(window)
            sys_bp_values.extend(window[sys_indices])
            dias_bp_values.extend(window[dias_indices])
            mean_bp_values.extend([np.mean(window)])
    return sys_bp_values, dias_bp_values, mean_bp_values


def mean_and_std(data, num_digits=1):
    return np.round(np.mean(data), num_digits), np.round(np.std(data), num_digits)


def min_and_max(data, num_digits=1):
    return np.round(np.min(data), num_digits), np.round(np.max(data), num_digits)


def get_patient_ids(files):
    return set([str(utils.get_patient_from_file(s)) for s in files])


def get_nibp_stats():
    with open(os.path.join(project_configs.stats_file_dir, "nibp_stats.txt"), "w") as nibp_f:
        file_sets = dict()
        file_sets["TRAIN"] = get_patient_ids(glob.glob(os.path.join(project_configs.train_dir, "*")))
        file_sets["VAL"] = get_patient_ids(glob.glob(os.path.join(project_configs.val_dir, "*")))
        file_sets["TEST"] = get_patient_ids(glob.glob(os.path.join(project_configs.test_dir, "*")))

        for name, patient_ids in file_sets.items():

            nibp_f.write("=" * 50 + "\n" + "=" * 50 + "\n")
            nibp_f.write(name)
            nibp_f.write("\n" + "=" * 50 + "\n" + "=" * 50 + "\n")

            df = pd.DataFrame()
            nibp_files = glob.glob(os.path.join(project_configs.nibp_data_dir, "*.csv.gz"))
            for f in nibp_files:
                # f = nibp_files[0]
                pt_df = pd.read_csv(f, header=0)
                pt_df["charttime"] = pd.to_datetime(pt_df["charttime"])
                patient_id = os.path.basename(f).split("_")[0]
                pt_df["patient_id"] = patient_id
                df = df.append(pt_df)

            # get only patients in this set
            print(df.shape)
            df = df[df["patient_id"].isin(patient_ids)]
            print(df.shape)

            # remove outliers
            min_bp = 10
            max_bp = 300
            for var in df.columns.values[df.columns.str.startswith("NIBP")]:
                df = df[(df[var] >= min_bp) & (df[var] <= max_bp)]

            df["charttime"] = pd.to_datetime(df["charttime"])
            df["date"] = pd.to_datetime(df["charttime"]).dt.date
            df["date"] = pd.to_datetime(df["date"])

            nibp_f.write("Found {} patients\n".format(df['patient_id'].unique().shape[0]))

            # look at time difference between cuffs
            # group by day since there can be multiple days between admissions
            nibp_f.write(str(df.groupby([df["patient_id"], df["date"]]).apply(
                lambda x: x['charttime'] - x['charttime'].shift()).fillna(0).agg(
                ['mean', 'median', 'std', 'min', 'max'])))

            rounding_digits = 1
            # var = "NIBP_mean"
            for var in df.columns.values[df.columns.str.startswith("NIBP")]:
                nibp_f.write("\n" + "*" * 50 + "\n")
                nibp_f.write(var)
                nibp_f.write("\n" + "*" * 50 + "\n")
                # counts of values per patient
                nibp_f.write("Counts of values per patient\n")
                nibp_f.write(str(
                    df.groupby("patient_id")[var].count().agg(['mean', 'std', 'min', 'max', 'median']).round(
                        rounding_digits)))
                nibp_f.write("\n" + "=" * 30 + "\n")
                # value statistics
                nibp_f.write("Value Statistics\n")
                nibp_f.write(str(df[var].agg(['mean', 'std', 'min', 'max', 'median']).round(rounding_digits)))
                nibp_f.write("\n")


def plot_bp_hist(df, save_path, xlabel="Blood Pressure (mmHg)"):
    colors = ["blue", "orange", "green"]

    fig, ax = plt.subplots(df.shape[1], 1, figsize=(16, 16), sharex=True, sharey=True)
    # for each file
    for j in range(df.shape[1]):
        ax[j].hist(df.iloc[:, j], alpha=1.0, bins=20, density=True,
                   color=colors[j], label=df.columns.values[j])
        tick_spacing = 10
        loc = plticker.MultipleLocator(base=tick_spacing)  # this locator puts ticks at regular intervals
        ax[j].xaxis.set_major_locator(loc)
        ax[j].legend(loc="upper right", fontsize=project_configs.axis_label_font_size + 4)
        ax[j].tick_params(labelsize=project_configs.axis_tick_label_size + 4)
    plt.xlabel(xlabel, fontsize=project_configs.axis_label_font_size + 4)
    plt.ylabel("Fraction of population", fontsize=project_configs.axis_label_font_size + 4)
    plt.tight_layout()
    fig.show()
    fig.savefig(save_path)


def get_bp_stats(p=0.1):
    train_files = glob.glob(os.path.join(train_file_dir, "*.npy"))
    val_files = glob.glob(os.path.join(val_file_dir, "*.npy"))
    test_files = glob.glob(os.path.join(test_file_dir, "*.npy"))

    # randomly sample to speed up time to compute metrics
    train_files = random.sample(train_files, k=int(p * len(train_files)))
    val_files = random.sample(val_files, k=int(p * len(val_files)))
    test_files = random.sample(test_files, k=int(p * len(test_files)))

    # training dataset
    sys_bp_values_train, dias_bp_values_train, mean_bp_values_train = get_window_bp_vals(train_files)

    bp_vals = {"Systolic BP": sys_bp_values_train,
               "Diastolic BP": dias_bp_values_train,
               "Mean BP": mean_bp_values_train}
    with open(os.path.join(wave_stats_dir, "train_data_bp_stats.txt"), "w") as out_f:
        for k, v in bp_vals.items():
            out_f.write("{}: {} ({})\n".format(k, *mean_and_std(v)))
            out_f.write("Min-Max {}: {}-{}\n".format(k, *min_and_max(v)))
            out_f.write("Number of {} beats: {} (p was set to {})\n".format(k, int(len(v)*(1./p)), p))

    # validation dataset
    sys_bp_values_val, dias_bp_values_val, mean_bp_values_val = get_window_bp_vals(val_files)

    bp_vals = {"Systolic BP": sys_bp_values_val,
               "Diastolic BP": dias_bp_values_val,
               "Mean BP": mean_bp_values_val}
    with open(os.path.join(wave_stats_dir, "val_data_bp_stats.txt"), "w") as out_f:
        for k, v in bp_vals.items():
            out_f.write("{}: {} ({})\n".format(k, *mean_and_std(v)))
            out_f.write("Min-Max {}: {}-{}\n".format(k, *min_and_max(v)))
            out_f.write("Number of {} beats: {} (p was set to {})\n".format(k, int(len(v)*(1./p)), p))

    # testing dataset
    sys_bp_values_test, dias_bp_values_test, mean_bp_values_test = get_window_bp_vals(test_files)

    bp_vals = {"Systolic BP": sys_bp_values_test,
               "Diastolic BP": dias_bp_values_test,
               "Mean BP": mean_bp_values_test}
    with open(os.path.join(wave_stats_dir, "test_data_bp_stats.txt"), "w") as out_f:
        for k, v in bp_vals.items():
            out_f.write("{}: {} ({})\n".format(k, *mean_and_std(v)))
            out_f.write("Min-Max {}: {}-{}\n".format(k, *min_and_max(v)))
            out_f.write("Number of {} beats: {} (p was set to {})\n".format(k, int(len(v)*(1./p)), p))

    # overall dataset
    bp_vals = {"Systolic BP": sys_bp_values_train + sys_bp_values_val + sys_bp_values_test,
               "Diastolic BP": dias_bp_values_train + dias_bp_values_val + dias_bp_values_test,
               "Mean BP": mean_bp_values_train + mean_bp_values_val + mean_bp_values_test}
    with open(os.path.join(wave_stats_dir, "all_data_bp_stats.txt"), "w") as out_f:
        for k, v in bp_vals.items():
            out_f.write("{}: {} ({})\n".format(k, *mean_and_std(v)))
            out_f.write("Min-Max {}: {}-{}\n".format(k, *min_and_max(v)))
            out_f.write("Number of {} beats: {} (p was set to {})\n".format(k, int(len(v)*(1./p)), p))

    # create histogram plots for systolic, diastolic, and mean BP for all sets of patients
    bp_vals_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in bp_vals.items()]))
    bp_vals_df.to_csv(os.path.join(wave_stats_dir, "all_data_bp_vals.csv"), header=True, index=False)
    plot_bp_hist(bp_vals_df,
                 save_path=os.path.join(wave_stats_dir, "all_data_bp_hist.png"))

    # create histograms comparing train/val/test for systolic, diastolic, and mean BP
    # systolic BP
    bp_vals = {"Train": sys_bp_values_train, "Validation": sys_bp_values_val, "Test": sys_bp_values_test}
    bp_vals_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in bp_vals.items()]))
    bp_vals_df.to_csv(os.path.join(wave_stats_dir, "sys_bp_vals.csv"), header=True, index=False)
    plot_bp_hist(bp_vals_df,
                 save_path=os.path.join(wave_stats_dir, "sys_bp_hist.png"),
                 xlabel="Systolic Blood Pressure (mmHg)")
    # diastolic BP
    bp_vals = {"Train": dias_bp_values_train, "Validation": dias_bp_values_val, "Test": dias_bp_values_test}
    bp_vals_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in bp_vals.items()]))
    bp_vals_df.to_csv(os.path.join(wave_stats_dir, "dias_bp_vals.csv"), header=True, index=False)
    plot_bp_hist(bp_vals_df,
                 save_path=os.path.join(wave_stats_dir, "dias_bp_hist.png"),
                 xlabel="Diastolic Blood Pressure (mmHg)")
    # mean BP
    bp_vals = {"Train": mean_bp_values_train, "Validation": mean_bp_values_val, "Test": mean_bp_values_test}
    bp_vals_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in bp_vals.items()]))
    bp_vals_df.to_csv(os.path.join(wave_stats_dir, "mean_bp_vals.csv"), header=True, index=False)
    plot_bp_hist(bp_vals_df,
                 save_path=os.path.join(wave_stats_dir, "mean_bp_hist.png"),
                 xlabel="Mean Blood Pressure (mmHg)")


def get_window_stats():
    # get list of train/test patients
    file_sets = dict()
    file_sets["TRAIN"] = get_patient_ids(glob.glob(os.path.join(project_configs.train_dir, "*")))
    file_sets["VALIDATION"] = get_patient_ids(glob.glob(os.path.join(project_configs.val_dir, "*")))
    file_sets["TEST"] = get_patient_ids(glob.glob(os.path.join(project_configs.test_dir, "*")))
    file_sets["TOTAL"] = set.union(*[file_sets["TRAIN"], file_sets["VALIDATION"], file_sets["TEST"]])
    # get list of all wave stats files per patient
    wave_stats_files = glob.glob(os.path.join(wave_stats_dir, "*-*.csv")) + glob.glob(
        os.path.join(wave_stats_dir, "*-*.txt"))

    with open(os.path.join(wave_stats_dir, "window_stats.txt"), "w") as win_f:
        for name, patient_ids in file_sets.items():

            win_f.write("=" * 50 + "\n" + "=" * 50 + "\n")
            win_f.write(name)
            win_f.write("\n" + "=" * 50 + "\n" + "=" * 50 + "\n")

            # create giant data frame with all patients
            wave_stats_df = pd.DataFrame()
            for f in wave_stats_files:
                ws_df = pd.read_csv(f, sep=",", header=0)
                wave_stats_df = wave_stats_df.append(ws_df)

            # get patient ID from file name
            wave_stats_df["patient_id"] = wave_stats_df.iloc[:, 0].apply(
                lambda s: str(utils.get_patient_from_file(s)))

            # TODO: this works for count columns, but not for indexes/shapes
            # Group by patient and sum counts
            wave_stats_df = wave_stats_df.groupby("patient_id").sum()
            print("Found {} total patients".format(wave_stats_df.shape[0]))
            win_f.write("Found {} total patients\n".format(wave_stats_df.shape[0]))

            # calculate some stats
            wave_stats_df["valid_window_percent"] = wave_stats_df["valid_window_count"] / wave_stats_df[
                "total_window_count"] * 100.
            # number of minutes (seconds per window / 60 seconds per minute)
            wave_stats_df["total_minutes"] = wave_stats_df["total_window_count"] * window_length / 60.
            wave_stats_df["valid_minutes"] = wave_stats_df["valid_window_count"] * window_length / 60.

            # get number of patients that have no valid windows
            invalid_patients = wave_stats_df[wave_stats_df["valid_window_count"] == 0]
            print("Found {} patients with no valid windows".format(invalid_patients.shape[0]))
            win_f.write("Found {} patients with no valid windows\n".format(invalid_patients.shape[0]))

            # remove preprocessed files for patients under threshold for valid minutes
            invalid_patients = wave_stats_df[
                (wave_stats_df["valid_minutes"] < valid_minutes_threshold) & (wave_stats_df["valid_minutes"] > 0)]
            print("Found {} patients with < {} mins of valid data. Removing...".format(invalid_patients.shape[0],
                                                                                       valid_minutes_threshold))
            win_f.write(
                "Found {} patients with < {} mins of valid data. Removing...\n".format(invalid_patients.shape[0],
                                                                                       valid_minutes_threshold))

            num_removed = 0
            for i in invalid_patients.iloc[:, 0]:
                print(os.path.join(project_configs.train_dir, "p{:06d}*".format(i)))
                invalid_files = glob.glob(os.path.join(project_configs.train_dir, "p{:06d}*".format(i))) + \
                                glob.glob(os.path.join(project_configs.val_dir, "p{:06d}*".format(i))) + \
                                glob.glob(os.path.join(project_configs.test_dir, "p{:06d}*".format(i)))
                print(invalid_files)
                for invalid_file in invalid_files:
                    os.remove(invalid_file)
                    num_removed += 1
                print("=" * 30)
            print("Found and removed {} files".format(num_removed))
            win_f.write("Found and removed {} files\n".format(num_removed))

            # get patients in either train or test set
            wave_stats_df = wave_stats_df[wave_stats_df.index.isin(patient_ids)]
            print("Found {} {} patients".format(wave_stats_df.shape[0], name))
            win_f.write("Found {} {} patients\n".format(wave_stats_df.shape[0], name))

            # remove invalid patients from data frame
            wave_stats_df = wave_stats_df[wave_stats_df["valid_minutes"] >= 10]

            # print out wave stats
            print("=" * 30)
            print(wave_stats_df.describe(include="all"))
            print("=" * 30)
            print(wave_stats_df.median())
            print("" * 30)
            print(wave_stats_df[["total_window_count", "valid_window_count", "total_minutes", "valid_minutes"]].sum(
                axis=0))
            # write to file

            win_f.write("=" * 30 + "\n")
            win_f.write(str(wave_stats_df.describe(include="all")))
            win_f.write("\n" + "=" * 30 + "\n")
            win_f.write(str(wave_stats_df.median()))
            win_f.write("\n" + "=" * 30 + "\n")
            win_f.write(str(
                wave_stats_df[["total_window_count", "valid_window_count", "total_minutes", "valid_minutes"]].sum(
                    axis=0)))
            win_f.write("\n")


if __name__ == "__main__":
    if not os.path.exists(wave_stats_dir):
        os.makedirs(wave_stats_dir)

    print("*"*30)
    print("Getting window stats...")
    print("*" * 30)
    # get_window_stats()
    print("*" * 30)
    print("Getting NIBP stats...")
    print("*" * 30)
    # get_nibp_stats()
    print("*" * 30)
    print("Getting invasive blood pressure stats...")
    print("*" * 30)
    get_bp_stats(p=0.5)
    print("*" * 30)
    print("DONE")
    print("*" * 30)
