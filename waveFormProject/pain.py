# %%
'''
A module that data wrangles medication info coming from one dataframe and adding it into another. 
'''

__author__ = 'Simon Lee'

from asyncore import write
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
from multiprocessing import Process
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
    try:
        l = []
        for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients/bruh'):
            i = i[:18]
            l.append(i)
        
        if fileID not in l:
            print("skipped")
            return 0
        print("in")
        df = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/' + file)
    except EOFError:
        print("file failed")
        os.remove('/data2/mimic/ABP_with_Med_data/patients/' + file)
        return 0
        

if __name__ == "__main__":
    l = []
    for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients'):
        l.append(i)
    os.chdir('/data2/mimic/ABP_with_Med_data/patients')

    with Pool(8) as executor:
        results = executor.map(write_files, l)

#%%
'''
A module that data wrangles medication info coming from one dataframe and adding it into another. 
'''

__author__ = 'Simon Lee'

from asyncore import write
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

#mv_df = pd.read_csv(path1+file1, compression='gzip')
#mv_df2 = pd.read_csv(path1+file1, compression='gzip', index_col ="AMOUNTUOM")

#drug_df = pd.read_csv(path1+file2, compression='gzip')
#sample_df = pd.read_csv(path2+sample, compression='gzip')

import concurrent
from multiprocessing import Pool

def write_files(file):
    try:
        l = []
        for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients/'):
            l.append(i)
        
        if file in l:
            print("skipped")
            return 0
        print("good sign again")
        curr = pd.read_csv(file, compression='gzip')
        print("file read")
        curr['Epinephrine'] = 0
        curr['Dobutamine'] = 0
        curr['Dopamine'] = 0
        curr['Phenylephrine'] = 0
        curr['Norepinephrine'] = 0
        print("columns have been made")
        curr.to_csv('/data2/mimic/ABP_with_Med_data/patients/' + file ,compression='gzip')
        return 1
    except:
        return 0

if __name__ == "__main__":
    

    #df = pd.read_csv('/data2/mimic/ABP_with_Med_data/mg/mg_INPUTEVENTS_MV.csv.gz', compression='gzip')
    #df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(np.str)
    l = []
    for i in os.listdir(path2):
        l.append(i)
    

    os.chdir(path2)
    with Pool(10) as executor:
        
        results = executor.map(write_files, l[::-1])
    
        print(len(results))
        print(sum(results))
# %%
