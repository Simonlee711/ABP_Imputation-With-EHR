# simonlee
#%% 
print("hello")
# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
#%%
# import random patients waveform data
path = '/data2/mimic/mimic_preprocessed/'
os.chdir(path)
df = pd.read_csv('p097322-2161-09-01-16-28_merged.csv.gz', compression='gzip')
df.head(10)

# plot the time series of one of the pseudo_NIBP data
#fig, axs = plt.subplots(2, figsize = (10,2))
#fig.suptitle('Patient\'s p097322-2161 ekg and sp02 scan')
#axs[0].plot(df['Unnamed: 0'], df['ekg'])
#axs[1].plot(df['Unnamed: 0'], df['sp02'])

df_sub = df.loc[4000:4200]

plt.figure(figsize = (15,2))
plt.plot(df_sub['ekg'])
plt.title('patient\'s p097322-2161, ekg scan')
plt.show()

plt.figure(figsize = (15,2))
plt.plot(df_sub['sp02'], color = 'g')
plt.title('patient\'s p097322-2161, sp02 scan')
plt.show()




# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# basic functions to read in any csv file
def plot_ekg(dataframe, patientName):
    '''
    Uses matplotlib to plot ekg time series
    '''
    print(patientName)
    plt.figure(figsize = (15,2))
    plt.plot(dataframe['ekg'])
    #plt.title("patient\'s", patientName,", EKG scan")
    plt.show() 

def plot_sp02(dataframe, patientName):
    '''
    Uses matplotlib to plot SP02 time series
    '''
    plt.figure(figsize = (15,2))
    plt.plot(dataframe['sp02'], color = 'g')
    #plt.title('patient\'s', patientName,', SP02 scan')
    plt.show()

def read_csv_file(file_path, file):
    '''
    reads the contents of a csvfile
    '''
    df = pd.read_csv(file_path, compression='gzip')
    #df_sub = df[4000:4250]
    file = file.replace('_merged.csv.gz','')
    plot_ekg(df, file)
    plot_sp02(df, file)
#%%
  
# iterate through all file
counter = 0
for file in os.listdir():
    if counter == 5:
        break
    # Check whether file is in text format or not
    if file.endswith(".csv.gz"):
        file_path = f"{path}/{file}"
  
        # call read text file function
        read_csv_file(file_path, file)
        counter += 1

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

path = '/data2/mimic/mimic_preprocessed/'
os.chdir(path)

# basic functions to read in any csv file
def plot_ekg(dataframe, patientName):
    '''
    Uses matplotlib to plot ekg time series
    '''
    print(patientName)
    plt.figure(figsize = (15,2))
    plt.plot(dataframe['ekg'])
    #plt.title("patient\'s", patientName,", EKG scan")
    plt.show() 

def plot_sp02(dataframe, patientName):
    '''
    Uses matplotlib to plot SP02 time series
    '''
    plt.figure(figsize = (15,2))
    plt.plot(dataframe['sp02'], color = 'g')
    #plt.title('patient\'s', patientName,', SP02 scan')
    plt.show()

def read_csv_file(file_path, file):
    '''
    reads the contents of a csvfile
    '''
    df = pd.read_csv(file_path, compression='gzip')
    df_sub = df[2000:2250]
    file = file.replace('_merged.csv.gz','')
    plot_ekg(df_sub, file)
    plot_sp02(df_sub, file)

  
# iterate through all file
counter = 0
for file in os.listdir():
    if counter == 5:
        break
    # Check whether file is in text format or not
    if file.endswith(".csv.gz"):
        file_path = f"{path}/{file}"
  
        # call read text file function
        read_csv_file(file_path, file)
        counter += 1

# %%
from bz2 import compress
from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

path1 = '/data2/mimic/mimic-iii-clinical-database-1.4/'
path2 = '/data2/mimic/mimic_preprocessed'
path3 = '/data2/mimic/ABP_with_Med_data/'

file1 = 'INPUTEVENTS_MV.csv.gz'
file2 = 'DRGCODES.csv.gz' # this file contains codes but find the file with the actual drug names as well

output_path = '/data2/mimic/ABP_with_Med_data'

sample = '/p090959-2147-08-27-16-12_merged.csv.gz'

#%%
class DataWrangler():
    '''
    A class that accomplishes the scraping of CSV files and gets correct 
    medication info to corresponding patients
    '''
    def csv_parser(self, source, source2, printer=False):
        '''
        A method that parses csv files by drug units

        i.e. mg, /hour, g, etc.
        '''
        measurement = source['AMOUNTUOM'].unique()
        print(measurement)
        names = measurement
        path = '/data2/mimic/ABP_with_Med_data/mg/FileName.csv.gz'
        list_df = []
        counter = 0

        for i in range(len(list_df)):
            list_df[i]['SUBJECT_ID'] = list_df[i]['SUBJECT_ID'].astype(np.str)
            print(type(list_df[i]['SUBJECT_ID']))
        
        for i in measurement:
            if i == 'mg':
                temp = i
                i = i.replace('/', '.')
                names[counter] = source2.loc[temp]
                list_df.append(names[counter])
                p = Path(path).parent.joinpath(f"{i}_INPUTEVENTS_MV.csv.gz")
                names[counter].to_csv(p, compression='gzip')
                print(i, "is done")
                counter += 1
            else:
                continue

        # printing option flag
        if printer:
            for i in range(len(list_df)):
                display(list_df[i])

    def clean_csv(self, source, wave_path = path2, printer = False):
        '''
        A method that deletes patient info that is not within the waves directory
        '''
        source['SUBJECT_ID'] = source['SUBJECT_ID'].astype(np.str)
        patient_IDs = self.patient_identification(wave_path)
        df = df.loc[df['SUBJECT_ID'].isin(patient_IDs)]
        
        return df

    def clean_csv2(self, source, printer = False):
        '''
        A method that deletes patient info that is not within the waves directory
        '''
        drug_ID = self.drug_identification()
        df = df.loc[df['ITEM_ID'].isin(drug_ID)]
        return df
    
    def drug_identification(self):
        meds = ['ADRENACLICK',
                'ADRENALIN',
                'ADYPHREN',
                'AKOVAZ',
                'ANA-KIT',
                'ANAPHYLAXIS',
                'ANGIOTENSIN',
                'ARAMINE',
                'AUVI-Q',
                'BIORPHEN',
                'DOBUTAMINE',
                'DOBUTREX',
                'DOPAMINE',
                'DROXIDOPA',
                'EMERPHED',
                'EPHEDRINE',
                'EPINEPHRINE',
                'EPINEPHRINE-CHLORPHENIRAMINE',
                'EPINEPHRINE-DEXTROSE',
                'EPINEPHRINE-NACL',
                'EPINEPHRINESNAP-EMS',
                'EPINEPHRINESNAP-V',
                'EPIPEN',
                'EPISNAP',
                'EPY',
                'GIAPREZA',
                'LEVOPHED',
                'METARAMINOL',
                'MIDODRINE',
                'NEO-SYNEPHRINE',
                'NOREPINEPHRINE',
                'NOREPINEPHRINE-DEXTROSE',
                'NOREPINEPHRINE-SODIUM',
                'NORTHERA',
                'ORVATEN',
                'PHENYLEPHRINE',
                'PHENYLEPHRINE-LIDOCAINE',
                'PROAMATINE',
                'SYMJEPI',
                'TWINJECT',
                'VAZCULEP']
        df = pd.read_csv('/data2/mimic/mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz', compression='gzip')
        df = df.loc[df['LABEL'].str.upper().isin(meds)]
        display(df)
        Drug_ID = df.ITEMID.unique()
        data = Drug_ID.tolist()
        data = data[7:]

        return data


    def time_calculator(self, start, end, drug_amt, printer = False):
        '''
        calculates the end and start time of the patients drug administration
        and calculates the rate in which the drug is administered
        '''
        d1 = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        d2 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")

        diff = d1- d2

        difference = d1.timestamp() - d2.timestamp()
        rate = drug_amt/float(difference * 1000)
        
        # if printing flag is activated
        if printer:
            print("time difference (Hr:Min:Sec):", diff)
            print("time difference (sec):", difference)
            print("drug administration rate:", rate)
        
        return rate
        
        # write out to wave data


    def patient_identification(self, source, printer = False):
        '''
        a data scraping method that gets all the patient ID's from a source directory
        '''
        os.chdir(source)
        patient_IDs = set()
        for filename in os.listdir(source):
            filename = filename[2:7]
            patient_IDs.add(str(filename))

        if printer:
            print(patient_IDs)
        
        return patient_IDs

    def save_csv(self, source, filename):
        '''
        A method that simply writes out the reduced information to a csv file
        '''
        source.to_csv(filename,compression='gzip')
        print('succesfully saved', filename)

    def write_to_wave(self, wave_data, start, stop, rate):
        '''
        After all is done write out to wave data files and save it to my repository
        '''
        try:
            start = start[11:]
            stop = stop[11:]
            print(start)
            path = '/data2/mimic/mimic_preprocessed/'
            df = pd.read_csv(path+wave_data, compression='gzip')
            display(df)
            timeList = df['Unnamed: 0'].tolist()
            finalList = []
            for i in timeList:
                i = i[10:19]
                finalList.append(i)
            print(finalList[0])
            print('done')
            startIndex = finalList.index(start)
            endIndex = finalList.index(stop)
            df['Unnamed: 0'][startIndex, endIndex] = rate
            display(df)

            return df

        except ValueError:
            return df
#%%
cleaner = DataWrangler()
os.chdir(path2)
counter = 0
for filename in os.listdir(path2):
    counter += 1
    try:
        
        if filename == 'p000160-2174-11-06-10-12_merged.csv.gz' or filename == 'p000188-2161-12-09-17-50_merged.csv.gz' or filename == 'p000188-2161-12-10-17-58_merged.csv.gz' or filename == 'p000188-2162-01-10-18-53_merged.csv.gz' or filename == 'p000333-2147-04-25-22-02_merged.csv.gz':
            print('skipped')
            continue
            
        print(counter)
        curr = pd.read_csv(filename, compression='gzip')
        curr['Epinephrine'] = 0
        curr['Dobutamine'] = 0
        curr['Dopamine'] = 0
        curr['Phenylephrine'] = 0
        curr['Norepinephrine'] = 0
        cleaner.save_csv(curr, '/data2/mimic/ABP_with_Med_data/patients/' + filename)
    except FileNotFoundError:
        continue
# %%
df = np.load('/data2/mimic/ABP_with_Med_data/patients/results2/npy/p095830-2101-02-19-07-07_preprocessed.npy')
print(df.shape)
# %%
