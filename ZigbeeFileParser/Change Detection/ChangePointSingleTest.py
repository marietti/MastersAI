from ChangePoint2 import change_detection_single
from pylab import *
import pandas as pd

df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Raw\\cc2531_sniffer_2_3_2021out_5_Seconds_Count.csv")
y = np.reshape(np.array(df["0xbfb8"].tolist()), (1, np.array(df["0xbfb8"].tolist()).size))
seed(1)
noise = np.random.normal(0, 1, y.size)
y = y + noise
alpha = .01
n = 50
k = 10
y = y + noise
index_offset = 0
count = 0
while count < 10:
    if index_offset >= y.size - (n * 2 + k - 1):
        index_offset = 0
        count = count + 1
    index_y = n * 2 + k - 1 + int(index_offset)
    print(size(y_read))
    print(index_offset)
    [s, _, _] = change_detection_single(y_read, n, k, alpha, 5)
    #print(s)
    index_offset = index_offset + 1
