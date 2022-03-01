import concurrent
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy.matlib
import cProfile
import pandas as pd
from ChangePoint2 import change_detection, Reverse
from pylab import *
from os.path import exists
import copy
from concurrent.futures import ProcessPoolExecutor as Executor


def main_thread(list_of_args):
    # from concurrent.futures import ThreadPoolExecutor as Executor  # to use threads
    with ThreadPoolExecutor(max_workers=2) as executor_worker:
        future1 = executor_worker.submit(worker, list_of_args)
        future2 = executor_worker.submit(worker2, list_of_args)
        val1, val2 = future1.result(), future2.result()
        final_score = val1 + val2
    numpy.savetxt(list_of_args[0], final_score, delimiter=",")
    print("Saved: " + list_of_args[0])


def worker(list_of_args):
    print("Processing: " + list_of_args[0] + " score2")
    noise = np.random.normal(0, 1, list_of_args[1].size)
    y = list_of_args[1] + noise
    noise = np.random.normal(0, 1, list_of_args[1].size)
    y = list_of_args[1] + noise
    [score1, _, _] = change_detection(y, list_of_args[2], list_of_args[3], list_of_args[4], list_of_args[5])
    print("Completed: " + list_of_args[0] + " score2")
    return score1


def worker2(list_of_args):
    print("Processing: " + list_of_args[0] + " score2")
    noise = np.random.normal(0, 1, list_of_args[1].size)
    y = list_of_args[1] + noise
    noise = np.random.normal(0, 1, list_of_args[1].size)
    y = list_of_args[1] + noise
    [score2, _, _] = change_detection(np.array(Reverse(y)), list_of_args[2], list_of_args[3], list_of_args[4], list_of_args[5])
    print("Completed: " + list_of_args[0] + " score2")
    return score2


seed(1)

seconds = "10"
sensor = "0xbfb8"
n_range = [10, 50, 100, 200, 500, 1000]
k_range = [1, 5, 10, 15, 20, 40, 60, 80, 100]
fold_range = [2, 5, 10]
alpha_range = [0, .001, .01, .1, .2]
index = 0
folder = "C:/Users/nickb/source/repos/MastersAI/ZigbeeFileParser/FileParsing/Training/ChangePoint/3_23_2021/" + seconds + "_Seconds/"
df = pd.read_csv(
    "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\3_23_2021out_" + seconds + "_Seconds_Count.csv")

y_input = np.reshape(np.array(df[sensor].tolist()), (1, np.array(df[sensor].tolist()).size))

queue = list()

for n in n_range:
    for k in k_range:
        if n > k:
            for alpha in alpha_range:
                for fold in fold_range:
                    file = seconds + "_second_" + str(alpha) + "_alpha_" + str(fold) + "_fold_" + str(n) + "_n_" + str(k) + "_k_final_score.csv"
                    path_to_file = folder + file
                    file_exists = exists(path_to_file)
                    if not file_exists:
                        queue.append([path_to_file, copy.deepcopy(y_input), n, k, alpha, fold])
                    else:
                        print("File Already Exsists: " + file)
with ThreadPoolExecutor(max_workers=8) as executor:
    furtureIteams = {executor.submit(main_thread, item): item for item in list(queue)}
