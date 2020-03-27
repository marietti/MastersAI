# Load libriaries and functions.
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.glm import fit

tfk = tf.keras
tf.keras.backend.set_floatx("float64")
import tensorflow_probability as tfp

tfd = tfp.distributions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

inputfile = '../../FileParsing/Raw/3_12_2020out'

# Define helper functions.
scaler = StandardScaler()
detector = IsolationForest(n_estimators=1000, behaviour="deprecated", contamination="auto", random_state=0)
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

# Load data and keep only first six months due to drift.
data = pd.read_csv(inputfile, usecols=lambda column: column not in ["datetime", "frame", "rawdata", "res", "framevers", "res.1"])

# Select columns and remove rows with missing values.
columns = ["intrapan", "ackreq", "framepending", "security", "type", "srcaddrmode",
           "dstaddrmode", "seqnum", "dstpanid", "dstaddr", "srcpanid", "srcaddr"]

# Scale data to zero mean and unit variance.
X_t = scaler.fit_transform(data)

# Remove outliers.
is_inlier = detector.fit_predict(X_t)
X_t = X_t[(is_inlier > 0), :]

# Restore frame.
dataset = pd.DataFrame(X_t, columns=columns)

# Select labels for inputs and outputs.
inputs = ["PT08.S1(CO)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "AH"]
outputs = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

# Define some hyperparameters.
n_epochs = 50
n_samples = dataset.shape[0]
n_batches = 10
batch_size = np.floor(n_samples / n_batches)
buffer_size = n_samples

# Define training and test data sizes.
n_train = int(0.7 * dataset.shape[0])

# Define dataset instance.
data = tf.data.Dataset.from_tensor_slices((dataset[inputs].values, dataset[outputs].values))
data = data.shuffle(n_samples, reshuffle_each_iteration=True)

# Define train and test data instances.
data_train = data.take(n_train).batch(batch_size).repeat(n_epochs)
data_test = data.skip(n_train).batch(1).repeat(n_epochs)

# Define prior for regularization.
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(len(outputs), dtype=tf.float64), scale=1.0),
                        reinterpreted_batch_ndims=1)

# Define model instance.
model = tfk.Sequential({tfk.layers.InputLayer(input_shape=(len(inputs),), name="input"),
                        tfk.layers.Dense(10, activation="relu", name="dense_1"),
                        tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(len(outputs)), activation=None,
                                         name="distribution_weights"), tfp.layers.MultivariateNormalTriL(len(outputs),
                                                                                                         activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                                                                                                             prior,
                                                                                                             weight=1 / n_batches),
                                                                                                         name="output")},
                       name="model")

# Compile model.model.
compile(optimizer="adam", loss=neg_log_likelihood)

# Run training session.model.
fit(data_train, epochs=n_epochs, validation_data=data_test, verbose=False)

# Describe model.
model.summary()

tfp.layers.DenseFlipout(10, activation="relu", name="dense_1")

# Predict.
samples = 500
iterations = 10
test_iterator = tf.compat.v1.data.make_one_shot_iterator(data_test)
X_true, Y_true, Y_pred = np.empty(shape=(samples, len(inputs))), np.empty(shape=(samples, len(outputs))), np.empty(
    shape=(samples, len(outputs), iterations))
for i in range(samples):
    features, labels = test_iterator.get_next()
    X_true[i, :] = features
    Y_true[i, :] = labels.numpy()
    for k in range(iterations):
        Y_pred[i, :, k] = model.predict(features)

# Calculate mean and standard deviation.
Y_pred_m = np.mean(Y_pred, axis=-1)
Y_pred_s = np.std(Y_pred, axis=-1)
