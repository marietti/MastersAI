from datetime import datetime
import pandas as pd

threshold = 1.0
range1 = 20.0

data_csv = pd.read_csv('C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\Data\\Calendar_120_Seconds_Count.csv')
fo = open('C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\Data\\final_score_Calendar_120_Seconds_Count_alpha_1_n_100_k_20.txt', "r")
line = fo.readline()
cp_score_data = line.split(",")
occupied = list()
index = 0
for i in cp_score_data:
    if float(i) > threshold:
        if float(i) < range1:
            occupied.insert(index, 1)
        else:
            occupied.insert(index, 2)
    else:
        occupied.insert(index, 0)
    index += 1
data_csv.insert(2, "occupied", occupied)
data_csv.to_csv('C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\Data\\Calendar_120_Seconds_Count_3.csv', index=False)
