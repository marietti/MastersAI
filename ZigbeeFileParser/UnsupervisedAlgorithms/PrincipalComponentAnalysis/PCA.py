import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, Conv1D

from UnsupervisedAlgorithms.PrincipalComponentAnalysis.PCAPlot import pca_std

comp = 10
inputfile = '../../FileParsing/No_ms/output4_5_Seconds_Count.csv'

model = Sequential()
layers = 1
units = 128

model.add(Dense(units, input_dim=comp, activation='relu'))
model.add(GaussianNoise(pca_std))
for i in range(layers):
    model.add(Dense(units, activation='relu'))
    model.add(GaussianNoise(pca_std))
    model.add(Dropout(0.1))
model.add(Dense(27133, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

train = pd.read_csv(inputfile)
train.drop(['datetime'], axis=1, inplace=True)
raw_data = train['len_sum'].values.astype('int32')
Y_train = raw_data
Y_train = tf.keras.utils.to_categorical(Y_train)
train.drop(['len_sum'], axis=1, inplace=True)
X_train = train.values.astype('int32')

scaler = StandardScaler()
scaler.fit(X_train)

pca = PCA(n_components=comp)
pca.fit(X_train)

X_sc_train = scaler.transform(X_train)
X_pca_train = pca.fit_transform(X_sc_train)
pca_std = np.std(X_pca_train)

model.fit(X_pca_train, Y_train, epochs=30, batch_size=20, validation_split=0.15, verbose=1)