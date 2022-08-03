#%%
import sys
import numpy as np
import pandas as pd
import os
import glob
import random
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
import multiprocessing as mp

from tqdm import tqdm
import pywt
from scipy.signal import find_peaks, filtfilt, firwin
from sklearn.preprocessing import StandardScaler, RobustScaler

#sys.path.append(os.path.join(os.environ["HOME"], "code/github/ABP_pred/"))
sys.path.append('/home/simonlee/waveFormProject/')
import project_configs_simon as project_configs_simon


def is_valid_ekg(sig, max_peaks_per_sec=4, min_peaks_per_sec=0.5, plot=False, distance=50, sample_freq=100, min_max_threshold=7., debug=False):
    """
    This function determines whether or not the EKG signal in a window is "valid"

    X: 1D array
    max_peaks_per_second: max number of peaks we should see in a 1 sec
    window. Default is 4 since 220/60 = 3.6667 which is approx 4
    min_peaks_per_second: min number of peaks we should see in a 1 sec
    window. Default is 1, but this may not be a valid assumption
    distance: minimum number of samples required between peaks
    """
    # from scipy.signal import find_peaks_cwt, daub
    reason = ""
    # if np.max(np.abs(sig)) > min_max_threshold:
    #     reason = "max(abs(sig)) > min_max_threshold"
    #     return False, reason
    # elif np.var(sig) < 1e-4:
    if np.var(sig) < 1e-4:
        reason = "var(sig) < 1e-4"
        return False, reason
    num_secs = sig.shape[0] / float(sample_freq)

    # use CWT to find peaks in EKG
    scales = np.arange(1, 2)
    dt = 1 / float(sample_freq)
    [coeffs, freqs] = pywt.cwt(sig, scales, "morl", dt)

    indices_peaks, props = find_peaks(-coeffs.T[:, 0], distance=36, threshold=(0.00001, 0.5), width=(1, 10),
                                      prominence=(0.005, None), wlen=10)

    if debug:
        print("=" * 30)
        print(indices_peaks)
        print(np.std(indices_peaks[1:] - indices_peaks[:-1]))
        print("prominences:", props["prominences"])
        print("left thresholds:", props["left_thresholds"])
        print("right thresholds:", props["right_thresholds"])
    if plot:
        plt.plot(sig)
        plt.scatter(indices_peaks, sig[indices_peaks], marker='*', s=60, c='r', label='r_peak')
    if len(indices_peaks) > (max_peaks_per_sec * num_secs):
        if debug:
            print("number of peaks larger than max allowed")
        reason = "number of peaks ({}) larger than max allowed".format(indices_peaks.shape[0])
        return False, reason
    if len(indices_peaks) < (min_peaks_per_sec * num_secs):
        if debug:
            print("number of peaks smaller than min allowed")
        reason = "number of peaks ({}) smaller than min allowed".format(indices_peaks.shape[0])
        return False, reason
    return indices_peaks, props


def is_valid_spo2(sig, max_peaks_per_sec=4, min_peaks_per_sec=0.5, plot=False, distance=36, sample_freq=100,
                  min_max_threshold=7., debug=False):
    """
    This function determines whether or not the arterial pressure signal in a window is "valid"

    X: 1D array
    max_peaks_per_second: max number of peaks we should see in a 1 sec
    window. Default is 4 since 220/60 = 3.6667 which is approx 4
    min_peaks_per_second: min number of peaks we should see in a 1 sec
    window. Default is 1, but this may not be a valid assumption
    distance: minimum number of samples required between peaks
    """
    # if np.max(np.abs(sig)) > min_max_threshold:
    #     reason = "np.max(np.abs(sig)) > min_max_threshold"
    #     return False, reason
    if np.var(sig) < 1e-4:
        reason = "np.var(sig) < 1e-4"
        return False, reason

    num_secs = sig.shape[0] / float(sample_freq)

    indices_peaks, props = find_peaks(sig, distance=distance, threshold=(0.00001, 0.5), prominence=(0.05, None))
    if debug:
        print("*" * 30)
        print(indices_peaks)
        print(np.std(indices_peaks[1:] - indices_peaks[:-1]))
        print("prominences:", props["prominences"])
        print("left thresholds:", props["left_thresholds"])
        print("right thresholds:", props["right_thresholds"])
    if plot:
        plt.plot(sig)
        plt.scatter(indices_peaks, sig[indices_peaks], marker='*', s=60, c='r', label='r_peak')
    if len(indices_peaks) > (max_peaks_per_sec * num_secs):
        reason = "number of peaks ({}) larger than max allowed ({})".format(len(indices_peaks), max_peaks_per_sec * num_secs)
        return False, reason
    if len(indices_peaks) < (min_peaks_per_sec * num_secs):
        reason = "number of peaks ({}) smaller than min allowed ({})".format(len(indices_peaks), min_peaks_per_sec * num_secs)
        return False, reason
    return indices_peaks, props


def is_valid_art(sig, max_peaks_per_sec=4, min_peaks_per_sec=0.5, plot=False, distance=40, max_consec_diff=50,
                 sample_freq=100):
    """
    This function determines whether or not the arterial pressure signal in a window is "valid"

    X: 1D array
    max_peaks_per_second: max number of peaks we should see in a 1 sec
    window. Default is 4 since 220/60 = 3.6667 which is approx 4
    min_peaks_per_second: min number of peaks we should see in a 1 sec
    window. Default is 1, but this may not be a valid assumption
    distance: minimum number of samples required between peaks
    """
    num_secs = sig.shape[0] / float(sample_freq)
    reason = ""
    if np.mean(sig) < 30.:
        reason = "mean(sig) < 30."
        return False, False, False, reason
    elif np.mean(sig) > 200:
        reason = "mean(sig) > 200"
        return False, False, False, reason
    elif np.max(sig) > 300:
        reason = "max(sig) > 300"
        return False, False, False, reason
    elif np.min(sig) < 20:
        reason = "min(sig) < 20"
        return False, False, False, reason
    elif np.max(sig) < 60:
        reason = "max(sig) < 60"
        return False, False, False, reason
    elif np.var(sig) < 80:
        reason = "var(sig) < 80"
        return False, False, False, reason
    elif np.var(sig) < 1e-2:
        reason = "var(sig) < 1e-2"
        return False, False, False, reason

    indices_max, props_max = find_peaks(sig, distance=distance, threshold=(None, 5.0), prominence=(20, None))
    indices_min, props_min = find_peaks(-sig, distance=distance, threshold=(None, 5.0), prominence=(10, None))

    peak_diffs = []
    try:
        if len(indices_max) > len(indices_min):
            peak_diffs = np.subtract(np.array(indices_max[1:]), np.array(indices_min))
        elif len(indices_min) > len(indices_max):
            peak_diffs = np.subtract(np.array(indices_max), np.array(indices_min[:-1]))
        else:
            peak_diffs = np.subtract(np.array(indices_max), np.array(indices_min))
    except ValueError:
        peak_diffs = [1000]
    #     print("peak_diffs:", peak_diffs)
    if plot:
        plt.plot(sig)
        plt.scatter(indices_max, sig[indices_max], marker='*', s=60, c='r', label='r_peak')
        plt.scatter(indices_min, sig[indices_min], marker='*', s=60, c='y', label='r_peak')
    # if len(indices_max) > (max_peaks_per_sec*num_secs) or \
    #            len(indices_max) < (min_peaks_per_sec*num_secs) or \
    if len(indices_max) == 0:
        reason = "did not find any sys BP values"
        return False, False, False, reason
    elif len(indices_min) == 0:
        reason = "did not find any dias BP values"
        return False, False, False, reason
    elif np.any(np.abs(np.diff(sig[indices_max])) > max_consec_diff):
        reason = "difference between two consecutive sys BP values was > {}".format(max_consec_diff)
        return False, False, False, reason
    elif np.any(np.abs(np.diff(sig[indices_min])) > max_consec_diff):
        reason = "difference between two consecutive dias BP values was > {}".format(max_consec_diff)
        return False, False, False, reason
    elif np.any(np.array(peak_diffs) > distance):
        reason = "some distance between a min and a max peak was greater than {}".format(distance)
        return False, False, False, reason
    return list(indices_max), list(indices_min), props_max, props_min


def is_valid_window(ekg, spo2, art, min_threshold=40, max_threshold=60, spo2_art_threshold=15, plot=False):
    """
    Checks to see if a window is valid by checking individual
    signals, and then making sure they align correctly
    """
    ekg_peaks, ekg_props = is_valid_ekg(ekg, plot=False)
    # print(type(ekg_peaks))
    # make sure ekg signal is clean
    reason = "No reason.."
    if ekg_peaks is not False:
        bp_max_indices, bp_min_indices, props_max, props_min = is_valid_art(art, plot=False)
        # print(art_peaks)
        # make sure art pressure signal is clean
        if bp_max_indices is not False:
            # if first peak is ekg
            if ekg_peaks[0] < bp_max_indices[0]:
                # if we have more ekg peaks than art
                if ekg_peaks.shape[0] > np.array(bp_max_indices).shape[0]:
                    # and signal at end of window
                    # if ekg_peaks[-1] > (ekg.shape[0] - 50):
                    ekg_peaks = ekg_peaks[:-1]

                    # if first peak is arterial pressure
            if bp_max_indices[0] < ekg_peaks[0]:
                # skip first arterial peak
                bp_max_indices = bp_max_indices[1:]

            # get difference between peaks
            try:
                peak_diff = bp_max_indices - ekg_peaks
            # unless we have different shapes, which means we don't
            # have peaks that match up
            except ValueError:
                pass

            # check if spo2 signal is valid
            spo2_peaks, spo2_props = is_valid_spo2(spo2, plot=False)
            if spo2_peaks is not False:
                try:
                    spo2_art_peak_diff = spo2_peaks - bp_max_indices
                except ValueError:
                    #                     pass
                    reason = "Number of SpO2 peaks ({}) different from number of ABP peaks ({})".format(
                        spo2_peaks.shape[0], len(bp_max_indices))
                    # return False, reason
                    # TODO: check to make sure this produces proper filtering
                    return True, reason

                # for development
                if plot:
                    plt.plot(ekg)
                    plt.scatter(ekg_peaks, ekg[ekg_peaks], marker='*', s=60, c='r', label='r_peak')
                    plt.plot(spo2)
                    plt.scatter(spo2_peaks, spo2[spo2_peaks], marker='*', s=60, c='r', label='r_peak')
                    plt.plot(art)
                    plt.scatter(art_peaks, art[art_peaks], marker='*', s=60, c='r', label='r_peak')
                    plt.show()
                # make sure differences between peaks within threshold of acceptance
                if True:
                    #                 if np.mean(peak_diff) <= max_threshold:
                    if True:
                        #                     if np.mean(peak_diff) >= min_threshold:
                        #                         if True:
                        if np.mean(np.abs(spo2_art_peak_diff)) <= spo2_art_threshold:
                            return True, reason
                        else:
                            reason = "mean(np.abs(spo2_art_peak_diff)) > spo2_art_threshold"
                    else:
                        reason = "mean(peak_diff) < min_threshold"
                else:
                    reason = "mean(peak_diff) > max_threshold"
            # else SpO2 is not valid
            else:
                reason = "SpO2 not valid:" + spo2_props
        # else ABP is not valid
        else:
            reason = "ABP not valid: " + props_min
    # else EKG is not valid
    else:
        reason = "EKG not valid: " + ekg_props
    return False, reason


def is_valid_nibp(nibp_sys, nibp_dias, nibp_mean):

    reason = ""
    if nibp_sys < nibp_dias:
        reason = "nibp_sys < nibp_dias ({} < {})".format(nibp_sys, nibp_dias)
        return False, reason
    elif nibp_sys < nibp_mean:
        reason = "nibp_sys < nibp_mean ({} < {})".format(nibp_sys, nibp_mean)
        return False, reason
    elif nibp_dias > nibp_mean:
        reason = "nibp_dias > nibp_mean ({} > {})".format(nibp_dias, nibp_mean)
        return False, reason
    elif nibp_sys <= 0:
        reason = "nibp_sys <= 0 ({})".format(nibp_sys)
        return False, reason
    elif nibp_dias <= 0:
        reason = "nibp_dias <= 0 ({})".format(nibp_dias)
        return False, reason
    elif nibp_mean <= 0:
        reason = "nibp_mean <= 0 ({})".format(nibp_mean)
        return False, reason
    else:
        return True, reason


def min_max_scale(X, scale_min, scale_max, x_min=None, x_max=None, num_samples=10000, window_size=400):
    """
    Similar to sklearn's MinMaxScaler, but allows you to
    specify the min/max values of the input data to scale by
    to avoid the effect of outliers

    :param numpy.array X: matrix to scale
    :param int scale_min:
    :param int scale_max:
    :param float x_min:
    :param float x_max:
    :param int num_samples:
    :param int window_size:
    :return numpy.array X_scaled: scaled matrix
    """
    max_vals = []
    min_vals = []
    for i in range(num_samples):
        idx = np.random.randint(0, X.shape[0]-window_size)
        sig_vals = X.iloc[idx:idx+window_size]
        max_vals.append(np.max(sig_vals))
        min_vals.append(np.min(sig_vals))
    x_max = np.median(max_vals)
    x_min = np.median(min_vals)
    print("x_max:", x_max)
    print("x_min:", x_min)
    # min/max scale
    X_std = (X - x_min) / (x_max - x_min)
    X_scaled = X_std * (scale_max - scale_min) + scale_min
    return X_scaled


def find_signal_shift(sig1, sig2, max_shift=400):
    """
    Uses numpy correlate function to slide one window
    across another, and check correlation at each point.
    We find the index of the shift that generates the
    largest correlation, and we use this index to determine
    how many samples to shift the signals to get them to
    properly overlap

    :param numpy.array sig1: this signal will be truncated and slid
    :param numpy.array sig2: this will be the reference signal
    :param int max_shift: maximum number of samples to shift signal left/right
    :return: (correlation coefficient, offset)
    :rtype: numpy.array, int
    """
    idx = int(sig1.shape[0] / 2)
    small_window_size = sig1.shape[0] - max_shift
    big_window_size = sig2.shape[0]

    s1 = sig1.iloc[idx - int(small_window_size / 2):idx + int(small_window_size / 2)]
    s2 = sig2
    cc = np.correlate(s1, s2, mode='valid')

    offset = int(((big_window_size - small_window_size) / 2) - np.argmax(cc))
    return cc, offset


def get_signal_start(signal, window_size=400):
    """
    Find the index where the valid signal begins
    :param signal: Pandas dataframe containing signal
    :param window_size: number of samples in evaluation window
    :return: index where the valid signal starts
    """
    for i in range(0, signal.shape[0], window_size):
        if np.all(signal.iloc[i:i+window_size, :].std() > 0):
            return i


def get_signal_end(signal, window_size=400):
    """
    Find the index where the valid signal ends
    :param signal: Pandas dataframe containing signal
    :param window_size: number of samples in evaluation window
    :return: index where the valid signal ends
    """
    signal = signal.iloc[::-1, :]
    for i in range(0, signal.shape[0], window_size):
        if np.all(signal.iloc[i:i+window_size, :].std() > 0):
            return signal.shape[0] - i


def proximity(series):
    """
    From https://stackoverflow.com/questions/37847150/pandas-getting-the-distance-to-the-row-used-to-fill-the-missing-na-values

    This function is for calculating the number of NaN samples since the most recent valid measurement
    Ex:
              x  prox
        0   NaN     0
        1   NaN     1
        2   NaN     2
        3   3.0     0
        4   NaN     1
        5   NaN     2
        6   NaN     3
        7   5.0     0
        8   NaN     1
        9   NaN     2
        10  NaN     3
    :param series: series with NaN measurements between valid measurements
    :return: series with number of samples since most recent valid measurement
    """
    groupby_idx = series.notnull().cumsum()
    groupby = series.groupby(groupby_idx)
    return groupby.apply(lambda x: pd.Series(range(len(x)))).values


def create_additional_features(input_merged, plot=False):
    """
    Creates additional features using the physiological waveforms (ECG and PPG) and
    the non-invasive blood pressure (NIBP) measurements. Wavelet transforms are used to
    highlight particular features in the ECG and PPG waveforms (e.g. T-wave, diacrotic notch).
    Statistics based on historical NIBP measurements are also added as features. Before
    augmenting the waveform dataframe, we trim off signal from the start/end of the record
    that is flat (i.e. likely recordings before the sensors are placed on the patient).

    :param DataFrame input_merged: rows are time, columns are (in this order) ECG, PPG, NIBP sys, NIBP dias, NIBP mean
    :return: DataFrame, start_index, end_index
    """
    simple_nibp_features = set(list(input_merged.columns.values))
    # use prior measurements as additional features
    periods = [5, 10, 15]
    for p in periods:
        for s in project_configs_simon.nibp_column_names:
            col_name_string = "{}_{}_{}".format("median", s, str(p))
            input_merged[col_name_string] = input_merged[s].rolling(p).median()

            col_name_string = "{}_{}_{}".format("std", s, str(p))
            input_merged[col_name_string] = input_merged[s].rolling(p).std()

    # add feature that measures the number of samples since the most recent NIBP value was sampled
    input_merged["prox"] = proximity(input_merged[project_configs_simon.nibp_column_names[-1]])
    # print(input_merged["ARTm"].dropna().index)

    derived_nibp_features = list(set(list(input_merged.columns.values)) - simple_nibp_features)
    # since non-invasive BP time may not exactly line up with invasive sampling time, find
    # closest time point for merging
    # impute missing non-invasive measurements by filling forward
    #                 wav = input_merged.fillna(method='ffill')
    #                 wav = input_merged.interpolate(method='spline', order=3, limit_direction='forward', axis=0)
    # wav = input_merged.interpolate(method='linear', limit_direction='forward', axis=0)
    wav = input_merged.fillna(method='ffill')
    # then, if anything is still null, fill with zero
    wav = wav.fillna(0)

    wav.rename(columns=project_configs_simon.signal_column_names, inplace=True)
    print("merged shape:", wav.shape)

    # trim off signal from start/end of record where they are likely hooking up sensors
    pretrimmed_shape = wav.shape[0]
    
    wav = wav[["ekg", "ppg", "prox"] + project_configs_simon.nibp_column_names + ["epi", "dob", "dop", "phe", "nor", "art"]]

    return wav


def filter_wave(data, cutoff, taps, btype, fs=2):
    # generates filtered waveform
    if btype == 'lowpass':
        b = firwin(taps, cutoff, window='hamming', fs=fs)
    elif btype == 'bandpass':
        b = firwin(taps, cutoff, window='hamming', pass_zero=False)
    elif btype == 'highpass':
        b = firwin(taps, cutoff, window='hamming', pass_zero=False)
    wav_filter = filtfilt(b, 1, data)
    return wav_filter


def filter_df(wave_df, taps=31, sample_rate=100):
    fmax = sample_rate / 2.
    filtertypes = {'exp1': {'btype': 'lowpass', 'cutoff': 45 / fmax, 'median_window': 100, 'median_thresh': 3}}

    wave_df_filtered = pd.DataFrame(index=wave_df.index)

    # cutoff frequency for each waveform
    feature_freq = {"ekg": 16.,
                    "ppg": 16.,
                    "art": 16.}

    #     for feature in wave_df.columns.values:
    for feature in feature_freq.keys():
        wave_df_filtered[feature] = filter_wave(wave_df[feature], feature_freq[feature], taps,
                                                filtertypes['exp1']['btype'], fs=sample_rate)

    return wave_df_filtered


def filter_valid_waveforms(patient_id):
    """
    :param string patient_id: string containing patient identifier
    """
    if not os.path.exists(project_configs_simon.window_save_dir):
        os.makedirs(project_configs_simon.window_save_dir)
    if not os.path.exists(project_configs_simon.invalid_window_images):
        os.makedirs(project_configs_simon.invalid_window_images)
    if not os.path.exists(project_configs_simon.valid_window_images):
        os.makedirs(project_configs_simon.valid_window_images)
    if not os.path.exists(project_configs_simon.stats_file_dir):
        os.makedirs(project_configs_simon.stats_file_dir)
    filtering_stats = {}

    # to make sure we don't try to process a record that's already been processed, we create a "lock" file
    # which contains the IDs of all records we've tried to process
    # read "lock" file
    lock_file = "record.lockfile"
    lock_file_ids = set()
    if os.path.exists(lock_file):
        with open(lock_file, "r") as lf:
            lock_file_ids = set(lf.read().splitlines())
    else:
        with open(lock_file, "w") as lf:
            lf.write(patient_id + "\n")

    print("lock file IDs:", lock_file_ids)
    # check to make sure this patient isn't in the training set
    train_files = glob.glob(os.path.join(project_configs_simon.window_save_dir, "*.npy"))
    train_ids = set([os.path.basename(f).split("_")[0] for f in train_files])
    # if this patient is in the training set, skip
    if patient_id in train_ids or patient_id in lock_file_ids:
        print("{} already exists in training set. Skipping...".format(patient_id))
        return
    else:
        with open(lock_file, "a+") as lf:
            lf.write(patient_id + "\n")
        print("reading csv")
        input_merged = pd.read_csv(os.path.join(project_configs_simon.preprocessed_data_dir, "{}_merged.csv.gz".format(patient_id)), sep=",",
                                compression="gzip")
        print("merged shape:", input_merged.shape)
        #display(input_merged)
        input_merged.rename(columns=project_configs_simon.signal_column_names, inplace=True)
        print("renamed")
        # low-pass filter signal to remove artifacts
        input_merged[["ekg", "ppg", "art"]] = filter_df(input_merged[["ekg", "ppg", "art"]],
                                                         sample_rate=project_configs_simon.sample_freq)
        print("first round of filters are goof")
        # trim flat signal from record and create additional features from signal
        pretrimmed_shape = input_merged.shape[0]
        wav = create_additional_features(input_merged)
        print(wav.shape)
        np.save(os.path.join(project_configs_simon.window_save_dir, "{}_preprocessed.npy".format(patient_id)), wav)
    return filtering_stats

#%%
if __name__ == "__main__":
    # configs
    shuffle = True
    multithread = False

    print("Loading data from: {}".format('/data2/mimic/mimic_preprocessed'))
    files = glob.glob(os.path.join('/data2/mimic/mimic_preprocessed/', "*.csv.gz"))

    # get patient IDs from file names
    patient_ids = [os.path.basename(f).split("_")[0] for f in files]
    patient_ids = patient_ids[100:200]
    if shuffle:
        # shuffle the patient ID list
        random.shuffle(patient_ids)

        # iterate through all patients
    for p in patient_ids:
        print(p)
        filter_valid_waveforms(p)
# %%
