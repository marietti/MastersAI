import pandas as pd
from pylab import *

n_range = [100, 50, 25, 15, 10, 5]
k_range = [20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
fold_range = [10, 9, 8, 7, 6, 5, 4, 3, 2]
alpha_range = [.8, .7, .6, .5, .2, .1, .05, .01, .001]

df = pd.read_csv(
    "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\Change Detection\\Best_out_6.csv")

#best_percent best_zeros n k alpha fold true_positive true_negative false_negative false_positive ones_count zeros_count filename
data = df[["n", "k", "alpha", "fold", "true_positive", "true_negative", "false_negative", "false_positive", "ones_count" > 0, "zeros_count" > 0]]



time_stamps = np.reshape(np.array(df["datetime"].tolist()), (1, np.array(df["datetime"].tolist()).size))