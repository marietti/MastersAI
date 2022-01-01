import os
import pandas as pd
from pylab import *
from datetime import datetime, timedelta

threshold = 1.0
accuracy = 0
false_positive = 0
false_negative = 0
true_positive = 0
ones_count = 0
zeros_count = 0
occupied = 0
seconds = "5"

#run = False
run = True

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
alpha = .001
fold = 0
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
for subdir, dirs, files in os.walk(folder):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".csv"):
            best_percent = 0.0
            best_zeros = 0
            df = pd.read_csv(filepath)
            t = df.values.tolist()
            if filepath.find("_k_") > 0:
                val = filepath[filepath.find("_n_")+3:filepath.find("_k_")]
                k = int(val)
            else:
                k = 20
            if filepath.find("_n_") > 0:
                val = filepath[filepath.find("_fold_")+6:filepath.find("_n_")]
                n = int(val)
            else:
                n = 100
            if filepath.find("_alpha_") > 0:
                val = filepath[filepath.find("_second_")+8:filepath.find("_alpha_")]
                alpha = float(val)
            else:
                alpha = .001
            if filepath.find("_fold_") > 0:
                val = filepath[filepath.find("_alpha_")+7:filepath.find("_fold_")]
                fold = int(val)
            else:
                fold = 100
            zero = n + k
            flat_list = [item for sublist in t for item in sublist]
            zeros_score = zeros(zero)
            input_changepoint = np.concatenate((zeros_score, np.array(flat_list)))
            input_changepoint = input_changepoint[start_index:end_index]
            changpoint_index = 0
            total = 0
            correct = 0
            false_positive = 0
            false_negative = 0
            true_positive = 0
            true_negative = 0
            ones_count = 0
            zeros_count = 0
            occupied_track = 0
            input_occupied = occupied[start_index:end_index]
            for changepoint in input_changepoint:
                total += 1
                if changpoint_index < len(occupied):
                    if changepoint > threshold:
                        if occupied_track == 0:
                            occupied_track = 1
                        if input_occupied[changpoint_index] == 1 and occupied_track == 1:
                            correct += 1
                            true_positive += 1
                        else:
                            false_positive += 1
                        ones_count += 1
                    else:
                        if occupied_track == 1:
                            occupied_track = 0
                        if input_occupied[changpoint_index] == 0 and occupied_track == 0:
                            correct += 1
                            true_negative += 1
                        else:
                            false_negative += 1
                        zeros_count += 1
                changpoint_index += 1
            percent = correct/total
            if best_percent_global < percent:
                best_percent_global = percent
                print(best_percent_global)
                best_file_global = filename
                print(best_file_global)
                best_zeros_global = zero
                print(best_zeros_global)
            if best_percent < percent:
                best_percent = percent
                best_zeros = zero
            if ones_count > 0 and zeros_count > 0:
                best_record.append(tuple((percent, zero, n, k, alpha, fold, true_positive/ones_count, true_negative/zeros_count, false_negative/zeros_count, false_positive/ones_count, ones_count, zeros_count, filename)))
            elif ones_count == 0:
                best_record.append(tuple((percent, zero, n, k, alpha, fold, 0, true_negative/zeros_count, false_negative/zeros_count, 0, ones_count, zeros_count, filename)))
            elif zeros_count == 0:
                best_record.append(tuple((percent, zero, n, k, alpha, fold, true_positive/ones_count, 0, 0, false_positive/ones_count, ones_count, zeros_count, filename)))
f = open("Best_out_3_23_2021", 'w', newline='', encoding='utf-8')
f.write(
    "best_percent, best_zeros, n, k, alpha, fold, true_positive, true_negative, false_negative, false_positive, ones_count, zeros_count, filename\n")
for entry in best_record:
    f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}\n'.format(str(entry[0]), str(entry[1]),
                                                                                          str(entry[2]), str(entry[3]),
                                                                                          str(entry[4]), str(entry[5]),
                                                                                          str(entry[6]), str(entry[7]),
                                                                                          str(entry[8]), str(entry[9]),
                                                                                          str(entry[10]),
                                                                                          str(entry[11]),
                                                                                          str(entry[12])))
f.close()

