#check scikit-learn version
import sklearn

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

print(sklearn.__version__)


# get the dataset
def get_dataset():
    df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\No_ms\\Calendar_11_15_9_1_5_Seconds_Count.csv")
    X = df[['0xd7b2', 'len_sum', 'smallest', 'largest', 'count_sum']]
    y = df['occupied']
#    X, y = make_classification(n_samples=1000, n_features=15, n_informative=10, n_redundant=5, random_state=6)
    return X, y


# get a list of models to evaluate
def get_models():
    df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\No_ms\\Calendar_11_15_9_1_5_Seconds_Count.csv")
    X = df[['0xd7b2', 'len_sum', 'smallest', 'largest', 'count_sum']]
    y = df['occupied']
    models = dict()
    # explore depths from 1 to 10
    for i in range(1, 11):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
        # fit the model on the whole dataset
        models[str(i)].fit(X, y)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X, y)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

