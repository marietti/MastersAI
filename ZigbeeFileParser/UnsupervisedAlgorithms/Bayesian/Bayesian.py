import numpy as np
import pandas as pd
import tensorflow as tp
import inferpy as inf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation

N = 10

M = 1000
dim_z = 10
dim_x = 100

#Define the decoder network
input_z = keras.layers.Input(input_dim=dim_z)
layer = keras.layers.Dense(256, activation='relu')(input_z)
output_x = keras.layers.Dense(dim_x)(layer)
decoder_nn = keras.models.Model(inputs=input, outputs = output_x)

#define the generative model
with inf.replicate(size=N):
    z = np.Normal(0, 1, dim=dim_z)
    x = np.Bernoulli(logits=decoder_nn(z.value()), observed = True)

#define the encoder network
input_x = keras.layers.Input(input_dim=d_x)
layer = keras.layers.Dense(256, activation='relu')(input_x)
output_loc = keras.layers.Dense(dim_z)(layer)
output_scale = keras.layers.Dense(dim_z, activation='softplus')(layer)
encoder_loc = keras.models.Model(inputs=input, outputs=output_mu)
encoder_scale = keras.models.Model(inputs=input, outputs=output_scale)

#define the Q distribution
q_z = tp.Normal(loc=encoder_loc(x.value()), scale=encoder_scale(x.value()))

#compile and fit the model with training data
inf.probmodel.compile(infMethod='KLqp', Q={z: q_z})
inf.probmodel.fit(x_train)

#extract the hidden representation from a set of observations
hidden_encoding = inf.probmodel.predict(x_pred, targetvar=z)
