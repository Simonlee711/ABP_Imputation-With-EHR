import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import reduce
import subprocess as sp
import argparse

from tqdm import tqdm
import wfdb
from scipy.signal import butter, filtfilt, freqz, firwin, freqz
from scipy import signal
from scipy.signal import find_peaks

sys.path.append("/home/arudas/projects/Waveforms/ABP_pred")
#sys.path.append("~/projects/Waveforms/ABP_pred/src")
import src
import src.project_configs as project_configs
from src.features.build_features import is_valid_art, filter_df
import src.waveform_helpers as waveform_helpers


def add_sample_to_dirname(data_dir, dir_name):
    if os.path.isdir(os.path.join(data_dir, dir_name)):
        return os.path.join(dir_name, os.path.basename(os.path.dirname(dir_name)))
    else:
        return dir_name


def resample_wave(data, samp_freq, resamp_freq, start_date=None):
    """
    Resample a waveform to a different sample frequency

    :param data: pandas DataFrame containing signal
    :param samp_freq: frequency that the waveform was sampled at
    :param resamp_freq: freqncy to resample the wave to
    :param start_date: date to use for start of waveform record
    :return: resampled DataFrame
    """
    if start_date is None:
        start_date = data.index[0]
    columns = data.columns.values
    # pad with zeros to speed up resample
    n = data.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = int(np.power(2, y+1))
    pad_width = ((0, nextpow2-n), (0,0))
    print(pad_width)
    print('orig data shape:', data.shape)
    print(np.count_nonzero(~np.isnan(data)))
    # pad data to speed up FFT
    data = np.pad(data, pad_width=pad_width, mode='constant')
    print("data shape:", data.shape)

    # replace nan values with zero or we get all nan values
    data = np.nan_to_num(data)

    newshape = int(data.shape[0] * resamp_freq / samp_freq)
    print("new shape:", newshape)
    f = signal.resample(data, newshape)
    sf = int(1e9 / resamp_freq)
    wavetime = pd.date_range(start_date, periods=newshape, freq=str(sf) + 'N')
    f = pd.DataFrame(f, columns=columns)
    print(np.count_nonzero(~np.isnan(f)))
    f = f.set_index(wavetime)
    return f


def get_pseudo_nibp_values(df, aline_col="ABP", fs=100, nibp_sample_rate_mins=3, window_size=4):
    """
    aline_col: column name of arterial line data
    fs: sampling frequency
    nibp_sample_rate_mins: number of minutes between non-invasive blood pressure sampling
    window_size: window size in seconds
    """
    num_samples_between_nibp = int(fs*60*nibp_sample_rate_mins)
    window_size = int(window_size*fs)
    
    # create placeholder columns
    df["pseudo_NIBP_sys_{}min".format(str(nibp_sample_rate_mins))] = np.nan
    df["pseudo_NIBP_dias_{}min".format(str(nibp_sample_rate_mins))] = np.nan
    df["pseudo_NIBP_mean_{}min".format(str(nibp_sample_rate_mins))] = np.nan

    num_invalid = 0
    num_mismatch = 0

#     pbar = tqdm(range(num_samples_between_nibp, df.shape[0]-window_size, num_samples_between_nibp))
#     for i in pbar:
    for i in range(num_samples_between_nibp, df.shape[0]-window_size, num_samples_between_nibp):
        abp = df[aline_col].iloc[i-window_size:i]

        # get indices of sys/dias if available
        bp_max_indices, bp_min_indices, props_max, props_min = is_valid_art(abp.values)
        if bp_max_indices is not False:
            if len(bp_max_indices) != len(bp_min_indices):
                num_mismatch += 1
            sys_bp_values = abp.iloc[bp_max_indices]
            dias_bp_values = abp.iloc[bp_min_indices]
            df["pseudo_NIBP_sys_{}min".format(str(nibp_sample_rate_mins))].iloc[i] = np.median(sys_bp_values)
            df["pseudo_NIBP_dias_{}min".format(str(nibp_sample_rate_mins))].iloc[i] = np.median(dias_bp_values)
            df["pseudo_NIBP_mean_{}min".format(str(nibp_sample_rate_mins))].iloc[i] = np.mean(abp)
        else:
            num_invalid += 1

    print("number invalid: {}".format(num_invalid))
    print("number mismatch: {}".format(num_mismatch))
    return df


def get_patient_id_as_int(patient_id):
    """
    Turns a MIMIC patient ID string into an integer we can use to query the database
    example: mimic3wdb/matched/p00/p000107/p000107-2122-05-14-21-01  --> 107
    :param patient_id: string containing the patient ID
    :return: int identifier
    """
    ptid = os.path.basename(patient_id).split("-")[0]
    # try to cast this to an int
    try:
        ptid = int(ptid)
    # except some patient IDs start with a character, so remove and try again
    except ValueError:
        try:
            ptid = int(ptid[1:])
        except ValueError as e:
            print(e)
            print("Guilty ID: {}".format(ptid))
            exit()
    return ptid


def get_nibp_values(df, sample_id, data_dir):
    # get patient ID as integer
    patient_id = get_patient_id_as_int(sample_id)
    file_name = os.path.join(data_dir, "{}_NIBP.csv.gz".format(patient_id))
    if not os.path.exists(file_name):
        print("{} does not exist! Skipping...".format(file_name))
        return None
    # load NIBP values
    nibp_df = pd.read_csv(os.path.join(data_dir, "{}_NIBP.csv.gz".format(patient_id)), sep=",", header=0, index_col=0)
    # make sure indices are datetime
    nibp_df.index = pd.to_datetime(nibp_df.index)
    df.index = pd.to_datetime(df.index)

    # merge on time indices
    df = pd.merge_asof(left=df, right=nibp_df,
                       left_index=True, right_index=True,
                       direction="backward",
                       tolerance=pd.Timedelta('2ms'))
    return df


def get_obs_file(patient_id, data_dir, data_type="obs"):
    """
    For UCI data
    This functions finds the path to the observation file for a given patient ID
    """
    id_suffix = patient_id.split("-")[-1]
    dir_path = sp.check_output(['find', data_dir, '-type', 'd', '-name', id_suffix + '*']).strip().decode("utf-8")
    print(dir_path)
    if not os.path.exists(dir_path):
        print("ERROR:", dir_path, "does not exist")
        return None
    if data_type == "obs":
        file_path = dir_path = sp.check_output(
            ['find', dir_path, '-name', 'Edwards Observation Data Extract*']).strip().decode("utf-8")
    elif data_type == "patient":
        file_path = dir_path = sp.check_output(
            ['find', dir_path, '-name', 'Edwards Patient Data Extract*']).strip().decode("utf-8")
    print(file_path)
    if not os.path.exists(file_path):
        print("ERROR:", file_path, "does not exist")
    else:
        return file_path

def get_ptpatient(patient_path):
    """
    For UCI data - get patient-specific data
    :param patient_path: path to patient data file
    :return: pandas DataFrame with patient data
    """
    try:
        patient_data = pd.read_csv(patient_path)
        patient_data['Anesthesia Start Time'] = pd.to_datetime(patient_data['Anesthesia Start Time'])
        patient_data['Anesthesia Stop'] = pd.to_datetime(patient_data['Anesthesia Stop'])
    except:
        print('No Patient data')
        patient_data = None
    return patient_data


def get_ptobs(observation_path):
    """
    For UCI data - get surgery observation data
    :param observation_path: path to observation file
    :return: pandas DataFrame with observation data
    """
    try:
        obs_data = pd.read_csv(observation_path)
        # conver time columns and sort in ascending
        obs_data['Captured D/T'] = pd.to_datetime(obs_data['Captured D/T'], format='%m/%d/%y   %H:%M')
        obs_data = obs_data.sort_values(['Captured D/T'], ascending=True)
        obs_data['Captured  Value'] = obs_data['Captured  Value'].astype(str)
        obs_data['Captured  Value'] = obs_data['Captured  Value'].map(lambda x: x.replace(',', ''))
        obs_data['Captured  Value'] = obs_data['Captured  Value'].astype(float)
        obs_data = obs_data.reset_index(drop=True)
    except:
        print('No Observation data')
        return None
    return obs_data


def get_bp_obs(obs_df):
    """
    For UCI data
    Takes the patient observation dataframe and gets the
    non-invasive blood pressure measurements in wide format
    """
    try:
        # get the non-invasive blood pressure measurements
        # bp_obs = obs_df[obs_df["Obs Type"].isin(["NIBP SYS", "NIBP DIA", "NIBP Mean"])]
        bp_obs = obs_df[obs_df["Obs Type"].isin(["ARTs", "ARTd", "ARTm"])]
        # pivot table from long format to wide format
        bp_obs_pivot = bp_obs.pivot(index="Captured D/T", columns="Obs Type", values="Captured  Value")
        # set index to be of type datetime
        bp_obs_pivot.index = pd.to_datetime(bp_obs_pivot.index)
        return bp_obs_pivot
    except:
        return None


def get_all_obs(obs_df):
    """
    For UCI data
    Takes the patient observation dataframe and gets the
    observation measurements in wide format
    """
    try:
        bp_obs = obs_df
        # pivot table from long format to wide format
        bp_obs_pivot = bp_obs.pivot(index="Captured D/T", columns="Obs Type", values="Captured  Value")
        # set index to be of type datetime
        bp_obs_pivot.index = pd.to_datetime(bp_obs_pivot.index)
        return bp_obs_pivot
    except:
        return None


def get_patient_waveforms(patient_id, path, ekg=True, spo2=True, art=True, min_ekg=-20, max_ekg=20,
                          min_spo2=-20, max_spo2=10, min_art=20, max_art=200, plot=False):
    """
    For UCI data - get ECG, PPG, and ART waveforms and join into a single file
    :param str patient_id: patient ID string for data we want to load

    :return: DataFrame containing all waveforms for patient
    :rtype: DataFrame
    """
    dataframes = []
    # EKG
    if ekg:
        patient_waveform = waveform_helpers.wavebin(ptid=patient_id, dataWavePath=os.path.join(path, "EKG/"))
        demo, freq, starttime, wave_length, wave = patient_waveform.read_bin_files()
        # TODO: CHECK THAT THIS ASSUMPTION IS VALID
        print("st:", starttime)
        print("wl:", wave_length)
        if (not isinstance(freq, float)) or (isinstance(starttime, list)):
            return None
        wavetime = patient_waveform.create_wavetime(starttime, len(wave), freq)
        ekg_df = pd.DataFrame(wave, columns=['wav'])
        ekg_df = ekg_df.set_index(wavetime)

        # downsample ekg to match sampling of art line and sp02
        print(ekg_df.shape)
        ekg_df = resample_wave(ekg_df, samp_freq=freq, resamp_freq=project_configs.sample_freq)
        print(ekg_df.shape)
        ekg_df.rename(index=str, columns={"wav": "ekg"}, inplace=True)
        # remove outlier values
        # ekg_df.loc[(ekg_df["ekg"] > max_ekg), "ekg"] = max_ekg
        # ekg_df.loc[(ekg_df["ekg"] < min_ekg), "ekg"] = min_ekg
        # scale signal to be between -1 and 1
        # ekg_df["ekg"] = min_max_scale(ekg_df["ekg"], scale_min=-1, scale_max=1, num_samples=10000)
        dataframes.append(ekg_df)

    # SpO2
    if spo2:
        patient_waveform = waveform_helpers.wavebin(ptid=patient_id, dataWavePath=os.path.join(path, "PLETH/"))
        demo, freq, starttime, wave_length, wave = patient_waveform.read_bin_files()
        print("st:", starttime)
        print("wl:", wave_length)
        assert (int(freq) == 100)
        wavetime = patient_waveform.create_wavetime(starttime, len(wave), freq)
        spo2_df = pd.DataFrame(wave, columns=['wav'])
        spo2_df = spo2_df.set_index(wavetime)
        spo2_df.rename(index=str, columns={"wav": "sp02"}, inplace=True)
        # remove outlier values
        # spo2_df.loc[spo2_df["sp02"] > max_spo2, "sp02"] = max_spo2
        # spo2_df.loc[spo2_df["sp02"] < min_spo2, "sp02"] = min_spo2
        # scale signal to be between -1 and 1
        # spo2_df["sp02"] = min_max_scale(spo2_df["sp02"], scale_min=-1, scale_max=1, num_samples=10000)
        dataframes.append(spo2_df)

    # Arterial Pressure
    if art:
        patient_waveform = waveform_helpers.wavebin(ptid=patient_id, dataWavePath=os.path.join(path, "ART/"))
        demo, freq, starttime, wave_length, wave = patient_waveform.read_bin_files()
        print("st:", starttime)
        print("wl:", wave_length)
        if freq == []:
            return None
        assert (int(freq) == 100)
        wavetime = patient_waveform.create_wavetime(starttime, len(wave), freq)
        art_df = pd.DataFrame(wave, columns=['wav'])
        art_df = art_df.set_index(wavetime)
        art_df.rename(index=str, columns={"wav": "art"}, inplace=True)
        # remove outlier values
        # art_df.loc[(art_df["art"] > max_art), "art"] = max_art
        # NOTE: should we set this to art_min or zero?
        # art_df.loc[(art_df["art"] < min_art), "art"] = min_art
        dataframes.append(art_df)

    # merge all signals into a single data frame
    wave_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="inner"),
                     dataframes)
    # convert index type to datetime
    wave_df.index = pd.to_datetime(wave_df.index)

    if plot:
        wave_df.plot()
        wave_df.show()
    return wave_df


def create_uci_waveform_files(sample_id, data_dir, save_dir, metadata_path="/data2/uci/UCIData/", debug=False, overwrite=False):
    """

    :param sample_id:
    :param data_dir:
    :param save_dir:
    :param metadata_path:
    :param debug:
    :param overwrite:
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_f = os.path.join(save_dir, "{}_merged.csv.gz".format(os.path.basename(sample_id)))
    # check if result file exists, and if so, only continue if we want to overwrite
    if os.path.exists(save_f) and overwrite is False:
        return None

    # get the non-invasive blood pressure
    nibp = get_all_obs(get_ptobs(get_obs_file(sample_id, metadata_path)))
    # get the patient-centric data
    pt = get_ptpatient(get_obs_file(sample_id, metadata_path, data_type="patient"))
    # make sure we have both data
    wav = None
    if nibp is not None and nibp.shape[0] > 0 and pt is not None:
        pt["Sex"] = pt["Sex"].apply(lambda x: 0 if str(x).lower() == "male" else 1)
        print(pt.head())
        nibp["Age"] = pt["DOB"].values[0]
        nibp["Height"] = pt["Ht (cm)"].values[0]
        nibp["Weight"] = pt["Wt (kg)"].values[0]
        nibp["Sex"] = pt["Sex"].values[0]
        print("nibp shape:", nibp.shape)
        print(nibp.head())
        # write NIBP stats to file
        # nibp_stats[patient_id] = nibp.shape[0]
        # pd.DataFrame.from_dict(nibp_stats, orient="index").to_csv("nibp_stats.csv", sep=",", index=True)

        # get the waveform data (ekg, spo2, art)
        wav = get_patient_waveforms(patient_id=sample_id, path=data_dir)
    if nibp is not None and pt is not None and wav is not None:

        # filter the wave
        # try:
        #     wav = filter_df(wav)
        # except ValueError:
        #     print(wav.shape)
        #     continue
        df = pd.merge_asof(wav, nibp, left_index=True, right_index=True, direction="nearest",
                           tolerance=pd.Timedelta('2ms'))
        # create pseudo-NIBP values from a-line
        df = get_pseudo_nibp_values(df, aline_col="art", fs=project_configs.sample_freq, nibp_sample_rate_mins=3)
        df = get_pseudo_nibp_values(df, aline_col="art", fs=project_configs.sample_freq, nibp_sample_rate_mins=5)
        # write to file
        df.to_csv(save_f, sep=",", header=True, index=True, compression="gzip")
        return df


def create_mimic_waveform_file(sample_id, data_dir, save_dir, 
    signals_of_interest=["II", "PLETH", "ABP"], 
    add_pseudo_nibp=True,
    add_nibp=True,
    debug=False, overwrite=False):
    min_ekg = -20
    max_ekg = 20
    min_spo2 = -20
    max_spo2 = 10
    min_art = 20
    max_art = 200
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_f = os.path.join(save_dir, "{}_merged.csv.gz".format(os.path.basename(sample_id)))
    # check if result file exists, and if so, only continue if we want to overwrite
    if os.path.exists(save_f) and overwrite is False:
        return None
    
    data_f = os.path.join(data_dir, sample_id)
    
    # read record
    try:
        record = wfdb.rdrecord(data_f, channel_names=signals_of_interest)
    except FileNotFoundError as e:
        print(e)
        return None
#     display(record.__dict__)
    print("p-signal shape:", record.p_signal.shape)
    
    # get indices of our signals of interest in record
    try:
        indices = [record.sig_name.index(sig) for sig in signals_of_interest]
    except ValueError as e:
        print(e)
        print("Skipping {}".format(sample_id))
        return None
    
    # create dataframe from record signals
    df = pd.DataFrame(record.p_signal[:, indices], columns=signals_of_interest)
    print("df shape:", df.shape)
    
    # trim record by removing null blocks at beginning and end
    first_idx = df.first_valid_index()
    last_idx = df.last_valid_index()
    print("first valid index: {} last valid index: {}".format(first_idx, last_idx))
    df = df.loc[first_idx:last_idx]
    
    if debug:
        print(df.describe())
        idx = np.random.randint(0, df.shape[0])
        win_length = 500
        fig, ax = plt.subplots(figsize=(12, 8))
        df.iloc[idx:idx+win_length, 0:2].plot(ax=ax)
        ax2 = ax.twinx()
        df.iloc[idx:idx+win_length, 2].plot(ax=ax2, color='g')
    
    # use record base time to create time index
    sf = int(1e9 / record.fs)
    start_date = pd.Timestamp(record.base_date.strftime("%Y-%m-%d") + " " + record.base_time.strftime("%H:%M:%S"))
    wavetime = pd.date_range(start_date, periods=df.shape[0], freq=str(sf) + 'N')
    df = df.set_index(wavetime)

    # resample wave down to 100 Hz
    df = resample_wave(df, samp_freq=record.fs, resamp_freq=project_configs.sample_freq, start_date=start_date)
    
    # fix outlier values
    # df.loc[(df["II"] > max_ekg), "II"] = max_ekg
    # df.loc[(df["II"] < min_ekg), "II"] = min_ekg
    # df.loc[(df["PLETH"] > max_spo2), "PLETH"] = max_spo2
    # df.loc[(df["PLETH"] < min_spo2), "PLETH"] = min_spo2
    # df.loc[(df["ABP"] > max_art), "ABP"] = max_art
    # df.loc[(df["ABP"] < min_art), "ABP"] = min_art

    # scale ECG and Pleth signals to be between -1 and 1
    #df["II"] = min_max_scale(df["II"], scale_min=-1, scale_max=1, num_samples=10000)
    #df["PLETH"] = min_max_scale(df["PLETH"], scale_min=-1, scale_max=1, num_samples=10000)
    
    # get pseudo-NIBP values
    if add_pseudo_nibp:
        df = get_pseudo_nibp_values(df, fs=project_configs.sample_freq, nibp_sample_rate_mins=3)
        df = get_pseudo_nibp_values(df, fs=project_configs.sample_freq, nibp_sample_rate_mins=5)
    if add_nibp:
        df = get_nibp_values(df, sample_id, project_configs.nibp_data_dir)

    # rename columns to match between datasets
    df.rename(columns=project_configs.signal_column_names, inplace=True)

    print(df.head())
    print(df.index)
    
    # save to file
    df.to_csv(save_f, sep=",", header=True, index=True, compression="gzip")
    
    return df


def main():

    parser = argparse.ArgumentParser(description="Creates .csv files\
     from the raw MIMIC waveform data")
    parser.add_argument('--record-file', 
        help='File containing MIMIC record names to use', 
        default="ECG_PLETH_ABP_IDs_wdb3_matched_subset.txt")
    parser.add_argument('--data-dir', 
        help='Directory containing the raw MIMIC data', 
        default=project_configs.raw_data_dir)
    parser.add_argument('--save-dir', 
        help='Directory to save output .csv files', 
        default=project_configs.preprocessed_data_dir)
    parser.add_argument('--signals', 
        help='List of signals of interest', 
        default=["II", "PLETH", "ABP"], 
        nargs='+')
    parser.add_argument('--skip-pseudo-nibp', 
        help='Flag to skip adding pseudo-NIBP',
        action='store_true')
    parser.add_argument('--skip-nibp', 
        help='Flag to skip adding NIBP',
        action='store_true')
    parser.add_argument('--num-process', 
        help='Number of processes to use (default: 1)', 
        default=1, type=int)
    args = parser.parse_args()


    record_f = args.record_file
    data_dir = args.data_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    signals = args.signals
    print("Using signals: ", signals)

    sample_list = [x.strip() for x in open(record_f, "r").readlines()]
    sample_list = [add_sample_to_dirname(data_dir, d) for d in sample_list]

    # print(sample_list)
    print("=" * 50)
    print("Found {} samples".format(len(sample_list)))
    print("=" * 50)

    random.shuffle(sample_list)

    if args.num_process > 1:
        print("Using {} processes in parallel...".format(args.num_process))
        Parallel(n_jobs=args.num_process)(
        delayed(create_uci_waveform_files)(sample_id, 
            data_dir=data_dir, 
            save_dir=save_dir) for sample_id in sample_list)
    else:
        for sample_id in sample_list:
            print(sample_id)
            create_uci_waveform_files(sample_id, 
                data_dir=data_dir, 
                save_dir=save_dir
                #signals_of_interest=signals, 
                #add_pseudo_nibp=not args.skip_pseudo_nibp, 
                #add_nibp=not args.skip_nibp
                )

    


if __name__ == "__main__":
    main()
