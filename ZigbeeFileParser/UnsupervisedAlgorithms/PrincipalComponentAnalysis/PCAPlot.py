import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.utils import np_utils

import matplotlib.pyplot as plt

train = pd.read_csv('../../FileParsing/No_ms/output4_5_Seconds_Count.csv')
#raw_data = pd.to_datetime(train['datetime'].values, format='%Y-%m-%dTH:%M:%S', errors='coerce')
train.drop(['datetime'], axis=1, inplace=True)
raw_data = train['len_sum'].values.astype('int32')
Y_train = raw_data
Y_train = np_utils.to_categorical(Y_train)
train.drop(['len_sum'], axis=1, inplace=True)
X_train = (train.values).astype('int32')

scaler = StandardScaler()
scaler.fit(X_train)
X_sc_train = scaler.transform(X_train)

pca = PCA(n_components=48)
pca.fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

plt.show()

scaler = StandardScaler()
scaler.fit(X_train)
X_sc_train = scaler.transform(X_train)
X_pca_train = pca.fit_transform(X_sc_train)
pca_std = np.std(X_pca_train)

print(X_sc_train.shape)
print(X_pca_train.shape)