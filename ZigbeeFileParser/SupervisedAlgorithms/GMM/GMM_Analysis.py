import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd

import os
folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, os.pardir, 'ZigbeeFileParser/FileParsing/Training/ChangePoint/2_3_2021/'))

plt.ion()
file = "/cc2531_sniffer_2_3_2021out_"

sensor = "0xbfb8"
sensor2 = "0xab52"
broadcast1 = "0x0000"
broadcast2 = "0xffff"

output = [[sensor, sensor2, broadcast1, broadcast2],[sensor, broadcast1, broadcast2], [sensor2, broadcast1, broadcast2], [sensor2], [sensor]]

df = pd.read_csv(folder + file + "5" + "_Seconds_Count.csv")

X = df[list([sensor, broadcast1, broadcast2]) + ['len_sum', 'smallest', 'largest', 'count_sum']]
y = df['occupied']

n_components = np.arange(100, 415)
models = [GMM(n, covariance_type='tied', random_state=0).fit(X)
          for n in n_components]

plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show(block=False)
plt.pause(300000)
