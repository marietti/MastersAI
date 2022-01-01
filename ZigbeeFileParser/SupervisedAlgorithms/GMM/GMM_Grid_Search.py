import sklearn

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot
import pandas as pd
import numpy as np

print(sklearn.__version__)
folder = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\2_3_2021\\"
folder_3 = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\"
file = "cc2531_sniffer_2_3_2021out_"
file_3 = "3_23_2021out_"

n_components = [3]
max_iter = [100, 200, 300]
covariance_type = ['full', 'tied', 'diag', 'spherical']

# get the dataset
def get_dataset():
    df = pd.read_csv(folder + file + "5_Seconds_Count.csv")
    X = df[list(z) + ['len_sum', 'smallest', 'largest', 'count_sum']]
    y = df['occupied']
    return X, y

# get a list of models to evaluate
def get_models():
    models = list()
    for n in n_components:
        # define ensemble model
        for m in max_iter:
            for c in covariance_type:
                models.append(tuple((GaussianMixture(n_components=n, max_iter=m, covariance_type=c),
                                     n, m, c)))
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=50, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

sensor = "0xbfb8"
sensor2 = "0xab52"
broadcast1 = "0x0000"
broadcast2 = "0xffff"
input = [sensor, sensor2, broadcast1, broadcast2]
output = [[sensor, sensor2, broadcast1, broadcast2],[sensor, broadcast1, broadcast2], [sensor2, broadcast1, broadcast2], [sensor2], [sensor]]

fo = open("Custom_Grid_Search_Out_Cross_Validation_GMM_3_Class.csv", 'w', newline='', encoding='utf-8')
for z in output:
    # define dataset
    X, y = get_dataset()
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for model in models:
        # evaluate the model
        scores = evaluate_model(model[0], X, y)
        fo.writelines(str(model[1]) + ',' + str(model[2]) + ',' + str(model[3]) + ',' + str(mean(scores)) + ',' + str(std(scores)) + '\n')
fo.close()
