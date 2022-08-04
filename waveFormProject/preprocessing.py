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

    l = []
    for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients/flags'):
        i = i[:18]
        l.append(i)
    
    if fileID not in l:
        print("skipped")
        return 0
    
    df = df.loc[df['date+ID'].isin([fileID])]
    #display(df)
    print("found something")
    df = df.reset_index()
    df_out = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/' + file ,compression='gzip')
    print("read in wave file")
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
            if len(df) >= 2 and i % 2 ==0:
                df_out.to_csv('/data2/mimic/ABP_with_Med_data/patients/' + file ,compression='gzip') 
                print("file successfully wrote out")  
        os.remove('/data2/mimic/ABP_with_Med_data/patients/flags/' + fileID + '.txt')
        return 1
    except IndexError:
        print("something failed")
        os.remove('/data2/mimic/ABP_with_Med_data/patients/flags/' + fileID + '.txt')
        return 0
        

if __name__ == "__main__":
    l = []
    for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients'):
        l.append(i)
    os.chdir('/data2/mimic/ABP_with_Med_data/patients')

    with Pool(6) as executor:
        results = executor.map(write_files, l[1000:3000])
# %%
def write_files(file, counter, member, f, f2, skip):
    fileID = file[:18]
    #print(fileID)

    l = []
    for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients/bruh'):
        i = i[:18]
        l.append(i)
    
    if fileID not in l:
        #print(file, "skipped")
        skip += 1
        return counter, member, skip

    if counter < 192 and file not in member:
        member.append(file)
        counter += 1
        f2.write(file + '\n')
    else:
        f.write(file+ '\n')
        member.append(file)
    return counter, member, skip

def write_files2(file, counter, member, f, f2, skip):

    if file in member:
        skip += 1
        return counter, member, skip

    if counter < 844:
        counter += 1
        f2.write(file + '\n')
    else:
        f.write(file+ '\n')
    return counter, member, skip
    
f = open('/data2/mimic/ABP_with_Med_data/patients/train/training.txt', "w")
f2 = open('/data2/mimic/ABP_with_Med_data/patients/test/testing.txt', "w")

l = []
for i in os.listdir('/data2/mimic/ABP_with_Med_data/patients'):
    l.append(i)
os.chdir('/data2/mimic/ABP_with_Med_data/patients')
member = []
counter = 0
skip = 0
for i in range(len(l)):
    counter, member,skip = write_files(l[i], counter, member, f,f2, skip)
print(skip)
skip = 0
counter = 0
for i in range(len(l)):
    counter, member,skip = write_files2(l[i], counter, member, f,f2, skip)
print(skip)
f.close()
f2.close()

# %%
