from datetime import datetime
import pandas as pd

threshold = 1.0
timewindow = [5, 10, 30, 45, 60, 75, 90, 105, 120]
folder = 'C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\'

f = open(folder + "\\" + "ratio_out", "w+")
for i in timewindow:
    filename = str(i) + "_second_final_score"
    fo = open(folder + filename + ".csv", "r")
    count1 = 0
    count0 = 0
    index = 0
    for line in fo:
        if abs(float(line)) > threshold:
            count1 += 1
        else:
            count0 += 1
        index += 1
    f.writelines(filename + ".csv\n")
    print("Count: " + str(index))
    f.writelines("Count: " + str(index) + "\n")
    print("One count: " + str(count1))
    f.writelines("One count: " + str(count1) + "\n")
    print("Zero count: " + str(count0))
    f.writelines("Zero count: " + str(count0) + "\n")
    print("Zero Count/Count: " + str(count0/index))
    f.writelines("Zero Count/Count: " + str(count0/index) + "\n")
    fo.close()
f.close()
