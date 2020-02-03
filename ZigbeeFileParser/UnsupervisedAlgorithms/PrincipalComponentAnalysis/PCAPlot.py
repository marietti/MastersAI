import matplotlib
import numpy as np
import pandas as pd
from pasta.augment import inline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

train = pd.read_csv('../../FileParsing/No_ms/output4.csv')
raw_data = pd.to_datetime(train['datetime'].values, format='%Y-%m-%dTH:%M:%S', errors='coerce')
Y_train = raw_data
Y_train = np_utils.to_categorical(Y_train)
train.drop(['datetime'], axis=1, inplace=True)
X_train = (train.values).astype('int32')

scaler = StandardScaler()
scaler.fit(X_train)
X_sc_train = scaler.transform(X_train)

pca = PCA(n_components=2)
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