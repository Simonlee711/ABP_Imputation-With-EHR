import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.signal import filtfilt, firwin
from scipy import signal


class wavebin():
    def __init__(self, ptid=None, dataWavePath=None, dataH5Path=None, AllPatList=None, AllFeatures=None):
        self.dataWavePath    = dataWavePath
        self.dataH5Path      = dataH5Path
        self.AllPatList      = AllPatList
        self.AllFeatures     = AllFeatures
        self.ptid            = ptid

    def create_pythondatetime(self, matlabdatenum):
        python_datetime = datetime.fromordinal(int(matlabdatenum)) + timedelta(days=matlabdatenum%1) - timedelta(days = 366)
        return python_datetime

    def create_wavetime(self, wavestarttime, timelength, sf):
        # sf is the sampling frequency
        # convert sf into nanoseconds
        print("sf:", sf)
        sf = int(1e9 / sf)
        wavetime = pd.date_range(wavestarttime, periods=timelength, freq=str(sf)+'N')
        return wavetime

    def read_bin_files(self):
        print('Reading ', self.dataWavePath, ' ', self.ptid)
        try:
            f = open(self.dataWavePath + self.ptid + '.bin', "rb")
            wave = np.fromfile(f, dtype=float)
            demo = wave[0:4]
            freq = wave[5]
            starttime = wave[6]
            starttime = self.create_pythondatetime(starttime)
            wave = wave[7:]
            f.close()
            return demo, freq, starttime, len(wave), wave
        except:
            return [], [], [], [], []
    def generate_hdf5_files(self):
        print('Generating Raw Wave HD5 files: ', self.ptid)
        fname = self.dataH5Path + self.ptid + '.h5'
        # if overwrite == False:
        if os.path.isfile(fname):
            print(fname+'H5 file already generated')
            return
        demo, freq, starttime, len_wave, wave = self.read_bin_files()
        if len(demo) > 0:
            # create a pandas datetime stamp for the wave
            wavetime = self.create_wavetime(starttime, len(wave), freq)
            df = pd.DataFrame(wave, columns=['wav'])
            df = df.set_index(wavetime)

            # store h5
            info = pd.DataFrame([freq], columns=['freq'])
            info['wavestarttime'] = starttime
            info['waveendtime'] = df.index[-1]
            print('Storing Raw Wave HD5 files: ', fname)
            store = pd.HDFStore(fname)
            store.put('raw_wave', df, format='table')
            store.put('info', info)
            store.close()


class wave_preprocess():
    def __init__(self, data):
        self.data     = data

    def filter_wave(self, cutoff, taps, btype):
        # generates filtered waveform
        if btype == 'lowpass':
            b = firwin(taps, cutoff, window='hamming')
        elif btype == 'bandpass':
            b = firwin(taps, cutoff, window='hamming', pass_zero=False)
        elif btype == 'highpass':
            b = firwin(taps, cutoff, window='hamming', pass_zero=False)
        wav_filter = filtfilt(b, 1, self.data['wav'])
        return wav_filter

    def resample_wave(self, samp_freq, resamp_freq):
        start_date = self.data.index[0]
        # pad with zeros to speed up resample
        n = self.data.shape[0]
        y = np.floor(np.log2(n))
        nextpow2  = int(np.power(2, y+1))
        pad_width = ((0, nextpow2-n), (0,0))
        print(pad_width)
        self.data = np.pad(self.data, pad_width=pad_width, mode='constant')
        
        newshape = int(self.data.shape[0] * resamp_freq / samp_freq)
        f = signal.resample(self.data, newshape)
        sf = int(1e9 / resamp_freq)
        wavetime = pd.date_range(start_date, periods=newshape, freq=str(sf) + 'N')
        f = pd.DataFrame(f, columns=['wav'])
        f = f.set_index(wavetime)
        return f

    def generate_filtered_wave(self, sf, taps, fmax, filtertype):
        # Filter requirements.
        # sf is the sampling frequency in Hz
        # filtertype is a dictionary of filtertypes with the name of the filter etc., filter type, and cutoffs
        wave_filter = self.filter_wave(filtertype['cutoff'], taps, filtertype['btype'])
        wave_filter = pd.DataFrame(wave_filter, columns=['filtered_wave'])
        wave_filter = wave_filter.set_index(self.data.index)
        return wave_filter

