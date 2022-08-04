#%%
'''
small data mining code that will sort of show me all the csv files headers
'''
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np



def read_csv_file(file_path, file):
    '''
    reads the contents of a csvfile
    '''
    df = pd.read_csv(file_path, compression='gzip')
    file = file.replace('_merged.csv.gz','')
    display(df)

if __name__ == "__main__":
    path = '/data2/mimic/mimic-iii-clinical-database-1.4/'
    os.chdir(path)
    for file in os.listdir():
        if file.endswith(".txt"):
            continue

        print('File:', file)
        file_size = os.path.getsize(file)
        print("File Size is :", file_size, "bytes")
        if file_size > 999999999:
            continue
        # Check whether file is in csv.gz format or not
        if file.endswith(".csv.gz"):
            file_path = f"{path}/{file}"
    
            # call read text file function
            read_csv_file(file_path, file)

# %%
df = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/p000160-2174-11-06-10-12_merged.csv.gz', compression='gzip')
display(df)
# %%
