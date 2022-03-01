import os
import pandas as pd
from pylab import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

threshold = 1.0
accuracy = 0
seconds = "10"
alpha = "0.001"
fold = "5"
n_file = "20"
k_file = "10"

run = False

#run = True

if run:
    start = datetime.strptime("2021-02-03 10:00:00", '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime("2021-02-03 17:00:00", '%Y-%m-%d %H:%M:%S')
    occupied_windows = ["2021-02-03 10:35:00", "2021-02-03 11:30:00", "2021-02-03 11:56:00", "2021-02-03 12:52:00", "2021-02-03 01:05:00", "2021-02-03 15:09:00", "2021-02-03 15:20:00", "2021-02-03 16:45:00"]
    folder = "C:/Users/nickb/source/repos/MastersAI/ZigbeeFileParser/FileParsing/Training/ChangePoint/2_3_2021/" + seconds + "_Seconds/"
    df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\2_3_2021\\cc2531_sniffer_2_3_2021out_" + seconds + "_Seconds_Count.csv")
else:
    start = datetime.strptime("2021-03-23 11:30:00", '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime("2021-03-23 17:00:00", '%Y-%m-%d %H:%M:%S')
    occupied_windows = ["2021-03-23 11:34:00", "2021-03-23 11:58:00", "2021-03-23 12:00:00", "2021-03-23 12:50:00", "2021-03-23 12:51:00", "2021-03-23 12:54:00", "2021-03-23 13:06:00", "2021-03-23 16:30:00"]
    folder = "C:/Users/nickb/source/repos/MastersAI/ZigbeeFileParser/FileParsing/Training/ChangePoint/3_23_2021/" + seconds + "_Seconds/"
    df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\3_23_2021out_" + seconds + "_Seconds_Count.csv")

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
start_index = 0
end_index = 0
track_start = 0
track_end = 0

compare = datetime.strptime(occupied_windows[occupied_windows_index], '%Y-%m-%d %H:%M:%S')
for timestamp in time_stamps[0]:
    if datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') >= start and track_start == 0:
        track_start = 1
    if datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') >= end and track_end == 0:
        track_end = 1
    if track_end == 0:
        end_index += 1
    if track_start == 0:
        start_index += 1
    if datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') >= compare and occupied_windows_index < len(occupied_windows):
        if not occupied_track == 0:
            occupied_track = 0
        else:
            occupied_track = 1
        occupied_windows_index += 1
        if occupied_windows_index < len(occupied_windows):
            compare = datetime.strptime(occupied_windows[occupied_windows_index], '%Y-%m-%d %H:%M:%S')
    occupied.append(occupied_track)

file = folder + seconds + "_second_" + alpha + "_alpha_" + fold + "_fold_" + n_file + "_n_" + k_file + "_k_final_score.csv"
df = pd.read_csv(file)
t = df.values.tolist()
if file.find("_k_") > 0:
    val = file[file.find("_n_") + 3:file.find("_k_")]
    k = int(val)
else:
    k = 20
if file.find("_n_") > 0:
    val = file[file.find("_fold_") + 6:file.find("_n_")]
    n = int(val)
else:
    n = 100
zero = n + k
flat_list = [item for sublist in t for item in sublist]
zeros_score = zeros(zero)
input_changepoint = np.array(flat_list)
input_changepoint = np.concatenate((zeros_score, np.array(flat_list)))
time = time_stamps[0]
zero_length = len(time) - len(input_changepoint)
print(zero_length)
zeros_score = zeros(zero_length)
input_changepoint = np.concatenate((input_changepoint, zeros_score))
clamp = lambda input_changepoint, minn, maxn: max(min(maxn, input_changepoint), minn)
plt.plot(time[start_index:end_index], input_changepoint[start_index:end_index], label="line 1")

zero_length = len(time) - len(occupied)
zeros_score = zeros(zero_length)
plt.plot(time[start_index:end_index], occupied[start_index:end_index], label = "line 2")
show()
