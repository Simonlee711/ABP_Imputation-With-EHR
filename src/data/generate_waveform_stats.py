import os
import sys
import glob
import multiprocessing as mp
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, firwin
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

sys.path.append("../../")
import src.project_configs as project_configs
import src.utils as utils
import src.utils as waveform_utils


art_col = project_configs.signal_column_names["ART"]
ecg_col = project_configs.signal_column_names["II"]
ppg_col = project_configs.signal_column_names["Pleth"]

mean_art_spectrum = np.load(open("art_mean_spectrum.npy", "rb"))
mean_ecg_spectrum = np.load(open("ecg_mean_spectrum.npy", "rb"))
mean_ppg_spectrum = np.load(open("ppg_mean_spectrum.npy", "rb"))

# functions for calculating window stats
stats_functions = {
    # art-related functions
    "mean(art)": lambda x: np.mean(x[art_col]),
    "std(art)": lambda x: np.std(x[art_col]),
    "min(art)": lambda x: np.min(x[art_col]),
    "max(art)": lambda x: np.max(x[art_col]),
    "max(art)-min(art)": lambda x: np.max(x[art_col]) - np.min(x[art_col]),
    "std(sliding mean(art))": lambda x: np.std(sliding_window_calc(x[art_col], np.mean)),
    "std(sliding std(art))": lambda x: np.std(sliding_window_calc(x[art_col], np.std)),
    "pulse_rate(art)": lambda x: calc_PULSE_RATE(x[art_col]),
    "pulse_variance(art)": lambda x: calc_PULSE_VARIANCE(x[art_col]),
    "PPV(art)": lambda x: calc_PPV(x[art_col]),
#     "std(sys)": lambda x: calc_SYS(x[art_col]),
#     "std(dias)": lambda x: calc_DIAS(x[art_col]),
    # ecg-related functions
#     "mean(ecg)": lambda x: np.mean(x[ecg_col]),
#     "std(ecg)": lambda x: np.std(x[ecg_col]),
    "min(ecg)": lambda x: np.min(x[ecg_col]),
    "max(ecg)": lambda x: np.max(x[ecg_col]),
    "std(sliding mean(ecg))": lambda x: np.std(sliding_window_calc(x[ecg_col], np.mean)),
    "std(sliding std(ecg))": lambda x: np.std(sliding_window_calc(x[ecg_col], np.std)),
#     "max(ecg)-min(ecg)": lambda x: np.max(x[ecg_col]) - np.min(x[ecg_col]),
#     # ppg-related functions
#     "mean(ppg)": lambda x: np.mean(x[ppg_col]),
#     "std(ppg)": lambda x: np.std(x[ppg_col]),
    "min(ppg)": lambda x: np.min(x[ppg_col]),
    "max(ppg)": lambda x: np.max(x[ppg_col]),
    "std(sliding mean(ppg))": lambda x: np.std(sliding_window_calc(x[ppg_col], np.mean)),
    "std(sliding std(ppg))": lambda x: np.std(sliding_window_calc(x[ppg_col], np.std)),
#     "max(ppg)-min(ppg)": lambda x: np.max(x[ppg_col]) - np.min(x[ppg_col]),
    "spectrum(art)": lambda x: get_spectrum_loss(x[art_col], mean_art_spectrum),
    "spectrum(ecg)": lambda x: get_spectrum_loss(x[ecg_col], mean_ecg_spectrum),
    "spectrum(ppg)": lambda x: get_spectrum_loss(x[ppg_col], mean_ppg_spectrum),
}


def get_spectrum_loss(signal, mean_spectrum):
    return 1. - np.max(np.correlate(get_spectrum(signal), mean_spectrum, mode="same"))


def get_spectrum(signal, num_coeffs=50):
    # subtract mean to remove zero-freq component
    signal = signal - np.mean(signal)

    P = np.correlate(signal, signal, mode="full")
    # get absolute magnitude of FFT
    Q = np.abs(np.fft.rfft(P).real[0:num_coeffs].reshape(-1, 1).T)
    # normalize component magnitude
    R = Q / np.sum(Q)
    return R.T.ravel()


def sliding_window_calc(window, function, sliding_win_size=project_configs.sample_freq):
    """
    Applies function to a window using a sliding-window
    :param pandas.Series window: window to generate sliding windows from
    :param func function: function to apply to each sliding window
    :param sliding_win_size: number of samples in each sliding window
    :return: list containing result of applying function to each sliding window
    """
    sliding_win_size = int(sliding_win_size)
    return [function(window.iloc[i:i + sliding_win_size]) for i in range(0, window.shape[0], sliding_win_size)]


def calc_PULSE_RATE(sig, sample_rate=project_configs.sample_freq):
    """
    Calculate pulse rate (beats/min) of the arterial pressure waveform
    :param numpy.array sig: arterial blood pressure waveform
    :param int sample_rate: sample rate of waveform signal
    :return: float pulse rate (beats/min) of waveform
    """
    # get systolic and diastolic blood pressure indices
    sys_indices, dias_indices = waveform_utils.get_art_peaks(sig, distance=36)
    # get average number of samples between beats
    avg_diff = np.mean(np.ediff1d(sys_indices))
    # divide average number of samples between beats by samples per min to get beats/min
    return 1. / (avg_diff / float(sample_rate)) * 60.


def calc_PULSE_VARIANCE(sig, sample_rate=project_configs.sample_freq):
    """
    Calculate pulse rate variance of the arterial pressure waveform
    :param numpy.array sig: arterial blood pressure waveform
    :param int sample_rate: sample rate of waveform signal
    :return: float pulse rate variance of waveform
    """
    # get systolic and diastolic blood pressure indices
    sys_indices, dias_indices = waveform_utils.get_art_peaks(sig, distance=36)
    # get average number of samples between beats
    var_diff = np.std(np.ediff1d(sys_indices))
    # divide average number of samples between beats by samples per min to get beats/min
    return var_diff / float(sample_rate)


def calc_PPV(sig, dias_to_sys_threshold=20):
    """
    Calculate the variance of the pulse pressure in the arterial pressure waveform
    :param numpy.array sig: arterial blood pressure waveform
    :param int dias_to_sys_threshold: max number of samples between diastolic point and systolic point
    :return: float variation of pulse pressure in waveform
    """
    # get systolic and diastolic blood pressure indices
    sys_indices, dias_indices = waveform_utils.get_art_peaks(sig, distance=36)

    if len(sys_indices) == 0 or len(dias_indices) == 0:
        return np.nan

    # convert to numpy arrays
    sys_indices, dias_indices = np.array(sys_indices), np.array(dias_indices)

    # only keep diastolic points that occur before last systolic point
    dias_indices = dias_indices[dias_indices < sys_indices[-1]]

    # calculate the pulse pressure for each systolic/diastolic BP beat pair
    # for each diastolic value, find closest systolic value occurring after diastolic point.
    # if the distance is within the valid threshold, calculate pulse pressure by subtracting
    # diastolic value from systolic value
    pulse_pressure_values = []
    for idx in dias_indices:
        # get closest systolic value that occurs after the diastolic point
        closest_idx = sys_indices[sys_indices > idx][0]
        # if the time difference between the two points is less than threshold, add to list
        if closest_idx - idx <= dias_to_sys_threshold:
            pulse_pressure = sig[closest_idx] - sig[idx]
            pulse_pressure_values.append(pulse_pressure)

    # calculate variance of pulse pressure in window
    return np.var(pulse_pressure_values)


def calc_SYS(sig):
    """
    Calculate mean systolic blood pressure of the arterial pressure waveform
    :param numpy.array sig: arterial blood pressure waveform
    :return: float mean systolic blood pressure of waveform
    """
    # get systolic and diastolic blood pressure indices
    sys_indices, dias_indices = waveform_utils.get_art_peaks(sig, distance=36)
    return np.std(sig[sys_indices])


def calc_DIAS(sig):
    """
    Calculate mean diastolic blood pressure of the arterial pressure waveform
    :param numpy.array sig: arterial blood pressure waveform
    :return: float mean diastolic blood pressure of waveform
    """
    # get systolic and diastolic blood pressure indices
    sys_indices, dias_indices = waveform_utils.get_art_peaks(sig, distance=36)
    return np.std(sig[dias_indices])


def create_stats_df(f, save_dir, overwrite=False):
    basename = os.path.basename(f).split(".")[0]
    save_file = os.path.join(save_dir, "{}_wave_stats.csv.bz2".format(basename))

    if os.path.exists(save_file) and not overwrite:
        print("{} exists and overwrite is set to False. Skipping file...".format(save_file))
        return

    stats_results = {k: [] for k, v in stats_functions.items()}

    # df = pd.read_csv(f, index_col=0)
    # print(df.head())
    # # create RobustScaler objects for signals with variable units
    # ecg_scaler = RobustScaler()
    # ppg_scaler = RobustScaler()
    # # scale signal using history
    # ecg_scaler.fit(df[ecg_col].values.reshape(-1, 1))
    # ppg_scaler.fit(df[ppg_col].values.reshape(-1, 1))
    # del df

    try:
        for window in tqdm(pd.read_csv(f, chunksize=project_configs.window_size, index_col=0)):

            # # transform window using RobustScaler
            # window[ecg_col] = ecg_scaler.transform(window[ecg_col].values.reshape(-1, 1))
            # window[ppg_col] = ppg_scaler.transform(window[ppg_col].values.reshape(-1, 1))

            for stat, func in stats_functions.items():
                stats_results[stat].append(func(window))

        stats_df = pd.DataFrame.from_dict(stats_results, orient="columns")
        stats_df.to_csv(save_file, header=True, index=False)
    except EOFError as e:
        print(e)
        print("guilty file:", f)
        return 


def generate_per_window_stats(save_dir="../../reports/waveform_stats_files", overwrite=False, num_threads=1):
    """
    Creates a per-window stats file for each input preprocessed waveform file
    :param save_dir: path to directory to save resulting stats files
    :param overwrite: if True, overwrite existing files if they exist
    :param num_threads: number of files to process in parallel
    :return: None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocessed_files = glob.glob(os.path.join(project_configs.preprocessed_data_dir, "*.csv.*"))
    print("Found {} files".format(preprocessed_files))

    if int(num_threads) > 1:
        pool = mp.Pool(num_threads)
        print("using {} CPUs".format(num_threads))
        results = pool.starmap(create_stats_df, [(p, save_dir, overwrite) for p in preprocessed_files])
        pool.close()
    else:
        # iterate through all patients
        for p in preprocessed_files:
            print(p)
            create_stats_df(p, save_dir=save_dir)


def summarize_per_window_stats(save_dir, overwrite=False, agg_func=np.mean):
    """
    Creates summary statistics for per-patient distribution by using aggregation function
    on each patient's window statistics
    :param save_dir: directory to save summary stats file
    :param overwrite: if True, overwrite summary stats file if it exists
    :param agg_func: function for aggregating per-window stats for each patient
    :return: pandas.DataFrame containing per-patient summary stats
    """
    save_file = os.path.join(save_dir, "summary_stats.csv.bz2")
    if os.path.exists(save_file) and not overwrite:
        print("{} exists and overwrite set to False. Skipping summary...".format(save_file))

    stats_files = glob.glob(os.path.join(save_dir, "*_wave_stats.csv.*"))
    print("Found {} stats files".format(len(stats_files)))

    stats_series = []
    for f in tqdm(stats_files):
        df = pd.read_csv(f)
        print(df.head())
        # stats_series.append(df.apply(agg_func, axis=0).to_frame().transpose())
        stats_series.append(df)

    stats_df = pd.concat(stats_series)
    print(stats_df.describe(include="all", percentiles=[.01, .025, .25, .5, .75, .975, .99]))
    stats_df.describe(include="all", percentiles=[.01, .025, .25, .5, .75, .975, .99]).to_csv(
        save_file, header=True, index=True)
    return stats_df


def main():
    parser = argparse.ArgumentParser(description="Generates per-window and per-patient statistics for waveforms")
    parser.add_argument('--generate', help="Generate per-window stats files", action="store_true", default=False)
    parser.add_argument('--overwrite', help="Overwrite existing files? (default=False)",
                        action="store_true", default=False)
    parser.add_argument('--num_threads', help="Number of threads to use (default=1)", default=1, type=int)
    parser.add_argument('--summarize', help="Create summary stats using the per-window stats files",
                        action="store_true", default=False)
    args = parser.parse_args()

    save_dir = "../../reports/waveform_stats_files"

    if args.generate:
        generate_per_window_stats(save_dir=save_dir, overwrite=args.overwrite, num_threads=args.num_threads)

    if args.summarize:
        summarize_per_window_stats(save_dir=save_dir, overwrite=args.overwrite, agg_func=np.mean)


if __name__ == "__main__":
    main()
