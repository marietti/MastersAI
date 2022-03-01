import numpy.matlib
import cProfile
import pandas as pd
from ChangePoint2 import change_detection, Reverse
from pylab import *

seed(1)

seconds = "60"
sensor = "0xbfb8"
window_size = 600
n_range = [20, 10]
k_range = [10, 5]
fold_range = [10, 5, 3]
alpha_range = [0, .001, .01, .1]
index = 0
folder = "C:/Users/nickb/source/repos/MastersAI/ZigbeeFileParser/FileParsing/Training/ChangePoint/3_23_2021/" + seconds + "_Seconds/"
df = pd.read_csv(
    "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\3_23_2021out_" + seconds + "_Seconds_Count.csv")

y_input = np.reshape(np.array(df[sensor].tolist()), (1, np.array(df[sensor].tolist()).size))

for n in n_range:
    for k in k_range:
        if n > k:
            for alpha in alpha_range:
                for fold in fold_range:
                    noise = np.random.normal(0, 1, y_input.size)
                    y = y_input + noise
                    y_trans = y.conj().transpose()
                    y_shape = shape(y_trans)
                    y_axis = y_shape[0]

                    [score1, _, _] = change_detection(y, n, k, alpha, fold)
                    [score2, _, _] = change_detection(np.array(Reverse(y)), n, k, alpha, fold)

                    final_score = score1 + score2
                    numpy.savetxt(folder + seconds + "_second_" + str(alpha) + "_alpha_" + str(fold) + "_fold_" + str(n) + "_n_" + str(k) + "_k_final_score.csv", final_score, delimiter=",")