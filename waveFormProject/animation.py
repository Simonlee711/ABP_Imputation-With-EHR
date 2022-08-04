#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

print("reading")
df = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/p001855-2124-03-02-10-41_merged.csv.gz')
df = df[21200:21600]
l = []
for i in df['Unnamed: 0.1']:
    i = i[11:]
    l.append(i)
df['Unnamed: 0.1'] = l
df.reset_index
print("df is done being read")

display(df)
fig, axs = plt.subplots(3, figsize=(8, 4))
fig.suptitle('EKG, PPG, & Medication time series')
x = ['0','1','2','3']
med = np.zeros(400)
med[50:100] = 4.0667519166666668e-06
med[250:334] = 2.0000076e-06

axs[0].plot(df['Unnamed: 0.1'], df['ekg'], label='EKG')
axs[0].plot(df['Unnamed: 0.1'][50:100], df['ekg'][50:100])
axs[0].plot(df['Unnamed: 0.1'][250:334], df['ekg'][250:334], color='orange')
axs[0].legend(loc='upper right')
axs[0].xaxis.set_visible(False)
axs[1].plot(df['Unnamed: 0.1'], df['sp02'], label='PPG')
axs[1].plot(df['Unnamed: 0.1'][50:100], df['sp02'][50:100])
axs[1].plot(df['Unnamed: 0.1'][250:334], df['sp02'][250:334], color='orange')
axs[1].legend(loc='upper right')
axs[1].xaxis.set_visible(False)
axs[2].plot(df['Unnamed: 0.1'], med)
axs[2].plot(df['Unnamed: 0.1'][49:101], med[49:101], color='orange')
axs[2].plot(df['Unnamed: 0.1'][249:335], med[249:335], label='MEDICATION', color='orange')
axs[2].legend(loc='upper right')
ticks = [0, 100, 200, 300, 400]
axs[2].set_xticks(ticks)
axs[2].tick_params(labelrotation=10)
#plt.margins(0)
plt.show()


# %%
l = []
for i in os.listdir('/data2/mimic/ABP_with_Med_data/mg/mg'):
    i = i[:18]
    l.append(i)
print(len(l))
for i in l:
    f = open('/data2/mimic/ABP_with_Med_data/patients/bruh/'+ i + '.txt', "w")
    f.write("yay")
    f.close()



# %%
print("reading")
df = pd.read_csv('/data2/mimic/ABP_with_Med_data/patients/p072377-2150-05-22-16-41_merged.csv.gz')
df = df[0:600000]
l = []
for i in df['Unnamed: 0.1.1']:
    i = i[11:]
    l.append(i)
df['Unnamed: 0.1'] = l

display(df)
fig, axs = plt.subplots(3, figsize=(8, 5))
fig.suptitle('EKG, PPG, & Medication time series')


axs[0].plot(df['Unnamed: 0.1'], df['ekg'], label='EKG')
axs[0].legend(loc='upper right')
axs[0].xaxis.set_visible(False)
axs[1].plot(df['Unnamed: 0.1'], df['sp02'], label='PPG')
axs[1].legend(loc='upper right')
axs[1].xaxis.set_visible(False)
axs[2].plot(df['Unnamed: 0.1'], df['Norepinephrine'],)
axs[2].legend(loc='upper right')
axs[2].xaxis.set_visible(False)
plt.margins(0)
plt.show()
# %%
