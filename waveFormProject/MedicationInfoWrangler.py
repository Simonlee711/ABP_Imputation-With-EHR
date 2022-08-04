#%%
'''
A module that data wrangles medication info coming from one dataframe and adding it into another. 
'''

__author__ = 'Simon Lee'

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process

print("Number of processors: ", mp.cpu_count())

# prints full array size without truncation
np.set_printoptions(threshold=sys.maxsize)

# directory paths for easier usability

path1 = '/data2/mimic/mimic-iii-clinical-database-1.4/'
path2 = '/data2/mimic/mimic_preprocessed'
path3 = '/data2/mimic/ABP_with_Med_data/'

file1 = 'INPUTEVENTS_MV.csv.gz'
file2 = 'DRGCODES.csv.gz' # this file contains codes but find the file with the actual drug names as well

output_path = '/data2/mimic/ABP_with_Med_data'

sample = '/p090959-2147-08-27-16-12_merged.csv.gz'

# Extracting the features from the files

mv_df = pd.read_csv(path1+file1, compression='gzip')
#mv_df2 = pd.read_csv(path1+file1, compression='gzip', index_col ="AMOUNTUOM")

#drug_df = pd.read_csv(path1+file2, compression='gzip')
#sample_df = pd.read_csv(path2+sample, compression='gzip')

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
            df['Unnamed: 0'][startIndex:endIndex] = rate
            display(df)

            return df

        except ValueError:
            return df


#%%
baseDf = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz', compression='gzip')
display(baseDf)

#%%
df = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/p000188-2161-12-09-17-50_merged.csv.gz', compression='gzip')
display(df)
#%%
import concurrent
from multiprocessing import Pool

def time_calculator(start, end, drug_amt):
    '''
    calculates the end and start time of the patients drug administration
    and calculates the rate in which the drug is administered
    '''
    d1 = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")

    difference = d1.timestamp() - d2.timestamp()
    rate = drug_amt/float(difference * 1000)

    return rate

def write_files(file):
    
    df = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz', compression='gzip')
    fileID = file[:18]
    print(fileID)

    l = []
    for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients/flags'):
        i = i[:18]
        l.append(i)
    
    if fileID not in l:
        print("skipped")
        return 0
    
    df = df.loc[df['date+ID'].isin([fileID])]
    display(df)
    df = df.reset_index()
    df_out = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/' + file ,compression='gzip')
    print("read in wave file")
    df_out['Epinephrine'] = 0
    df_out['Dobutamine'] = 0
    df_out['Dopamine'] = 0
    df_out['Phenylephrine'] = 0
    df_out['Norepinephrine'] = 0
    try:
        for i in range(len(df)):
            s = df['STARTTIME'].at[i]
            st = df['ENDTIME'].at[i]
            amount = df['AMOUNT'].at[i]
            rate = time_calculator(s, st, amount)
            print(rate)
            value = s+'.000'
            value2 = st+'.000'
            beg = (df_out[df_out['Unnamed: 0.1'] == value].index.values)
            end = (df_out[df_out['Unnamed: 0.1'] == value2].index.values)
            if rate == 0:
                continue
            # insert drug info to right column
            if df['ITEMID'][i] == 221906:
                df_out['Norepinephrine'][beg[0]:end[0]] = rate
            elif df['ITEMID'][i] == 221749:
                df_out['Phenylephrine'][beg[0]:end[0]] = rate
            elif df['ITEMID'][i] == 221662:
                df_out['Dopamine'][beg[0]:end[0]] = rate
            elif df['ITEMID'][i] == 221653:
                df_out['Dobutamine'][beg[0]:end[0]] = rate
            elif df['ITEMID'][i] == 221289:
                df_out['Epinephrine'][beg[0]:end[0]] = rate
            else:
                continue
            print("columns have successfully been put in")    
            df_out.to_csv('/data2/mimic/ABP_with_Med_data/patients/' + file ,compression='gzip') 
            print("file successfully wrote out")     
            os.remove('/data2/mimic/ABP_with_Med_data/patients/flags/' + fileID + '.txt')
        return 1
    except IndexError:
        print("something failed")
        os.remove('/data2/mimic/ABP_with_Med_data/patients/flags/' + fileID + '.txt')
        return 0
        

if __name__ == "__main__":
    cleaner = DataWrangler()
    
    # 1. clean csv - only extract mg
    cleaner.csv_parser(mv_df, mv_df2, True)
    # 2. save cleared csv
    df = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz')
    
    file2 = 'mg_INPUTEVENTS_MV'

    source_path = "/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz"
    for i,chunk in enumerate(pd.read_csv(source_path, chunksize=20000, compression = 'gzip')):
        chunk.to_csv('/data2/mimic/ABP_with_Med_data/mg/mg/{}{}.csv.gz'.format(file2, i), index=False, compression = 'gzip')

    cleaner = DataWrangler()

    #cleaner.csv_parser(mv_df, mv_df2)
    for i in range(43):
        df = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg/mg/mg_INPUTEVENTS_MV{}.csv.gz'.format(i), compression='gzip')
        final = cleaner.clean_csv(df)
        done = cleaner.clean_csv2(final)
        cleaner.save_csv(done, '/data2/mimic/ABP_with_Med_data/mg/mg/mg_INPUTEVENTS_MV{}.csv.gz'.format(i))

    import pandas as pd
    import glob
    import os

    # setting the path for joining multiple files
    extension = 'csv.gz'
    all_filenames = [i for i in glob.glob('/data2/mimic/ABP_with_Med_data/mg/mg/mg_INPUTEVENTS_MV*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f, compression = 'gzip') for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz", compression = 'gzip') 
    
    # what
    fileID = filename[2:7]
    date = filename[8:18]
    id = []
    id.append(fileID)
    df1 = df.loc[df['SUBJECT_ID'].isin(id)]
    dateList = df1['STARTTIME'].tolist()
    finalDateList = []
    for i in dateList:
        i = i[0:10]
        finalDateList.append(i)

    print(finalDateList)
    display(df1)
    if date in finalDateList:
        index = finalDateList.index(date)
        print('writing medication data out')
        df2 = df1.iloc[index]
        rate = cleaner.time_calculator(df2['STARTTIME'],df2['ENDTIME'], df2['AMOUNT'] )
        print(rate)
        df_final = cleaner.write_to_wave(filename, df2['STARTTIME'],df2['ENDTIME'], rate, df2['ITEMID'])
    
    print("writing out to file")
    cleaner.save_csv(df_final, '/data2/mimic/ABP_with_Med_data/patients/' + filename)

    # 3. calculate time of dataframe
    df = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz')
    del df["Unnamed: 0.1"]
    del df["Unnamed: 0"]
    del df["Unnamed: 0.1.1"]
    del df["Unnamed: 0.1.1.1"]
    display(df)
    cleaner.save_csv(df, '/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz')
    # 4. iterate through all files and begin to write out drug info
    l = []
    for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients'):
        l.append(i)
    os.chdir('/data2/mimic/ABP_with_Med_data/patients')

    with Pool(12) as executor:
        results = executor.map(write_files, l)


#%%
################################### SCRATCH WORK #######################################
# display dataframes
df = [mv_df2, sample_df]
for i in df:
display(i)

# unique ID's
ID_unique = mv_df['SUBJECT_ID'].unique()
patient_ID = []

for i in ID_unique:
i = str(i).replace(str(i),'p0'+str(i))
patient_ID.append(i)



#%%
os.chdir(path2)
# search for patients ID's and if they are not there print them
counter = 0
for path in patient_ID:
# check if current path is a file
cmd='find ' + path +'*'
print(cmd)
print('--------------------------------')
run = os.system(cmd)
print(run)
print('\n\n')
counter += 1
if counter == 1000:
    break


# time calculator
#%%

def time_calc(start, end, drug):
'''
#takes the end and start and 
'''
d1 = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
d2 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")

diff = d1- d2
print(diff)

difference = d1.timestamp() - d2.timestamp()
print(difference)
rate = drug/float(difference * 1000)
print(rate)


for i in range(1000):
start = mv_df.at[i,'STARTTIME']
end = mv_df.at[i,'ENDTIME']
drug_amount = mv_df.at[i, 'AMOUNT']
print(start, end)
time_calc(start, end, drug_amount)

#%%
# CSV seperator by drug AMOUNTUOM

measurement = mv_df['AMOUNTUOM'].unique()
print(measurement)
names = measurement
list_df = []
counter = 0
for i in measurement:
names[counter] = mv_df2.loc[i]
list_df.append(names[counter])
counter += 1
#print(list_df)

for i in range(len(list_df)):
display(list_df[i])

#%%
# get patient ID's of the waveforms directory
os.chdir(path2)
import re
patient_IDs = []
for filename in os.listdir(path2):
filename = filename[2:7]
patient_IDs.append(filename)

patient_IDs = set(patient_IDs)
patient_IDs = list(patient_IDs)
print(patient_IDs)

#%%
# iterate through columns of patient ID's, pattern match and save them to specific dataframe

mv_df = pd.read_csv(path1+file1, compression='gzip')
display(mv_df)
counter = 0
for i in mv_df['SUBJECT_ID']:
if counter == 1000:
    break
if counter % 100 == 0:
    print("currently at batch:", counter)
if i in patient_IDs:
    continue
mv_df = mv_df.drop(counter)
counter += 1
display(mv_df)

#%%
df = pd.read_csv(path1+file1, compression='gzip')
print(df['SUBJECT_ID'].dtypes)
df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(np.int32)
print(df['SUBJECT_ID'].dtypes)
display(df)


#%%

if __name__ == "__main__":
print("yes")

# %%
#%%

if __name__ == "__main__":
cleaner = DataWrangler()
# divides and generates new dataframes based on amountuom
cleaner.csv_parser(mv_df, mv_df2)
os.chdir(path3)
# reduces data by only considering patients in our test and training set
fileNamez = []
for file in os.listdir(path3):
    df = pd.read_csv(file, compression='gzip')
    final = cleaner.clean_csv(df)
    cleaner.save_csv(final, file)
    filename = file.replace('.csv.gz','')
    fileNamez.append(filename)

#
# 
# 
# 

file2 = '.hour_INPUTEVENTS_MV'
file3 = 'ml.hr_INPUTEVENTS_MV'

source_path = "/data2/mimic/ABP_with_Med_data/mg_INPUTEVENTS_MV.csv.gz"
for i,chunk in enumerate(pd.read_csv(source_path, chunksize=20000, compression = 'gzip')):
    chunk.to_csv('/data2/mimic/ABP_with_Med_data/mg/{}{}.csv.gz'.format(file2, i), index=False, compression = 'gzip')

cleaner = DataWrangler()

#cleaner.csv_parser(mv_df, mv_df2)
#for i in range():
df = pd.read_csv('/data2/mimic/ABP_with_Med_data/ml.hr_INPUTEVENTS_MV.csv.gz'.format(i), compression='gzip')
final = cleaner.clean_csv(df)
cleaner.save_csv(final, '/data2/mimic/ABP_with_Med_data/ml.hr_INPUTEVENTS_MV.csv.gz'.format(i))

import pandas as pd
import glob
import os

# setting the path for joining multiple files
extension = 'csv.gz'
all_filenames = [i for i in glob.glob('/data2/mimic/ABP_with_Med_data/ml/ml_INPUTEVENTS_MV*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f, compression = 'gzip') for f in all_filenames ])
#export to csv
combined_csv.to_csv( "/data2/mimic/ABP_with_Med_data/ml_INPUTEVENTS_MV.csv.gz", compression = 'gzip') 
'''
# %%

def patient_identification(source, printer = False):
    '''
    #a data scraping method that gets all the patient ID's from a source directory
    '''
    os.chdir(source)
    patient_IDs = set()
    for filename in os.listdir(source):
        filename = filename[2:7]
        patient_IDs.add(filename)

    if printer:
        print(patient_IDs)
    
    return patient_IDs

def write_to_wave(ID, start, stop, rate):
    '''
    #After all is done write out to wave data files and save it to my repository
    '''


    

def time_calculator(start, end, drug_amt, index, printer = False):
    '''
    #calculates the end and start time of the patients drug administration
    #and calculates the rate in which the drug is administered
    '''
    # checks if end time is less than start time in case operation lasted between two days

    if int(end[0:2]) < int(start[0:2]):
        
        # work in progress
        real_time = int(end[0:2]) + 24
        end1 = str(real_time) + end[2:]
        print(type(end1))
        df.at[index, 'ENDTIME'] = str(end1)
    

    d1 = datetime.strptime(end, "%H:%M:%S")
    d2 = datetime.strptime(start, "%H:%M:%S")
    #print(d1)

    diff = d1- d2

    difference = d1.timestamp() - d2.timestamp()
    if difference < 0:
        print('error checker later gator')
        return None
    print(difference)
    rate = drug_amt/float(difference * 1000)
    return rate


df = pd.read_csv('/data2/mimic/ABP_with_Med_data/grams_INPUTEVENTS_MV.csv.gz', compression='gzip')
filename = 'grams_INPUTEVENTS_MV'
filepath = '/data2/mimic/mimic_preprocessed/'
wave = pd.read_csv('/data2/mimic/mimic_preprocessed/p061150-2179-12-23-11-05_merged.csv.gz', compression='gzip')
display(wave)

patients = patient_identification(path2)

for i in range(2):
    ID = df.at[i, 'SUBJECT_ID']
    if ID in patients:
        ID = 'p0' + ID
        wave_csv = 
        start = df.at[i,'STARTTIME']
        start = start[11:]
        end = df.at[i,'ENDTIME']
        end = end[11:]
        drug_amount = df.at[i, 'AMOUNT']
        rate = time_calculator(start, end, drug_amount, i, filename)
        write_to_wave(ID, start, end, rate, wave_csv)





# %%
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
Drug_ID = df.ITEMID.unique()
data = Drug_ID.tolist()
data = data[7:]
data = set(data)

#%%

# more data reduction
df1 = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg_INPUTEVENTS_MV.csv.gz', compression='gzip')
for i in df1[]
display(df)


# %%

def clean_csv(source, printer = False):
    '''
    #A method that deletes patient info that is not within the waves directory
    '''
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
    df2 = pd.read_csv('/data2/mimic/mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz', compression='gzip')
    df2 = df2.loc[df2['LABEL'].str.upper().isin(meds)]
    Drug_ID = df2.ITEMID.unique()
    Drug_ID = int(Drug_ID)
    Drug_ID = set(Drug_ID)
    print(Drug_ID)

    source['ITEMID'] = source['ITEMID'].astype(np.str)
    print(type(source['ITEMID']))
    counter = 0
    
    df = source[:1]
    counter = 0
    for i in source['ITEMID']:
        print(counter)
        if counter == len(source['ITEMID']) - 1:
            break
        counter += 1
        if i in Drug_ID:
            print(i, "is added")
            df = df.append(source.iloc[counter,])
        else:
            print(i, "is skipped")
            continue
    #df = df.drop(0)
    display(df)
    
    return df
def save_csv(source, filename):
    '''
    A method that simply writes out the reduced information to a csv file
    '''
    source.to_csv(filename,compression='gzip')
    print('succesfully saved', filename)

df = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg_INPUTEVENTS_MV.csv.gz')
df2 = clean_csv(df)
save_csv(df2, '/data2/mimic/ABP_with_Med_data/mg_INPUTEVENTS_MV.csv.gz')
display(df2)

# %%
path1 = '/data2/mimic/mimic-iii-clinical-database-1.4/INPUTEVENTS_MV.csv.gz'
df = pd.read_csv(path1, compression='gzip')
display(df)
# %%
'''