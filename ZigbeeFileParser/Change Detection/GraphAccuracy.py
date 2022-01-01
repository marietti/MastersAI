import os
import pandas as pd
from pylab import *
from datetime import datetime
import matplotlib.pyplot as plt

limit = 6
threshold = 1.0
accuracy = 0
seconds = "5"
occupied_windows = ["2021-02-03 10:35:00", "2021-02-03 11:30:00", "2021-02-03 11:56:00", "2021-02-03 12:52:00", "2021"
                    "-02-03 01:05:00", "2021-02-03 15:09:00", "2021-02-03 15:20:00", "2021-02-03 16:45:00"]
folder = "C:/Users/nickb/source/repos/MastersAI/ZigbeeFileParser/FileParsing/Training/ChangePoint/"
df = pd.read_csv(
    "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\cc2531_sniffer_2_3_2021_Trainingout_" + seconds + "_Seconds_Count.csv")
time_stamps = np.reshape(np.array(df["datetime"].tolist()), (1, np.array(df["datetime"].tolist()).size))

occupied_windows_index = 0
occupied_windows_datetime = datetime.strptime(occupied_windows[occupied_windows_index], '%Y-%m-%d %H:%M:%S')
occupied_track = 0
occupied = []
n = 100
k = 20
max = 2 * n - 2 + k
best_percent_global = 0.0
best_zeros_global = 0
best_file_global = ''
best_record = list()

compare = datetime.strptime(occupied_windows[occupied_windows_index], '%Y-%m-%d %H:%M:%S')
for timestamp in time_stamps[0]:
    if datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') >= compare and occupied_windows_index < len(occupied_windows):
        if not occupied_track == 0:
            occupied_track = 0
        else:
            occupied_track = 1
        occupied_windows_index += 1
        if occupied_windows_index < len(occupied_windows):
            compare = datetime.strptime(occupied_windows[occupied_windows_index], '%Y-%m-%d %H:%M:%S')
    occupied.append(occupied_track)
for subdir, dirs, files in os.walk(r'C:\Users\nickb\source\repos\MastersAI\ZigbeeFileParser\FileParsing\Training\ChangePoint\5_Seconds'):
    for filename in files:
        filepath = subdir + os.sep + filename
        df = pd.read_csv(filepath)
        t = df.values.tolist()
        if filepath.find("_k_") > 0:
            val = filepath[filepath.find("_n_") + 3:filepath.find("_k_")]
            k = int(val)
        else:
            k = 20
        if filepath.find("_n_") > 0:
            val = filepath[filepath.find("_fold_") + 6:filepath.find("_n_")]
            n = int(val)
        else:
            n = 100
        zero = n + k
        flat_list = [item for sublist in t for item in sublist]
        zeros_score = zeros(zero)
        input_changepoint = np.concatenate((zeros_score, np.array(flat_list)))
        time = time_stamps[0]
        time = time[:len(input_changepoint)]
        plt.plot(time, input_changepoint, label="line 1")

        occupied = occupied[:len(time)]
        plt.plot(time, occupied, label="line 2")
        show()
