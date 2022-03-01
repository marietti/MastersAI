# example of grid searching key hyperparameters for adaboost on a classification dataset
import itertools

from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas
import joblib
from sklearn.metrics import confusion_matrix

folder = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\Change Point\\2_3_2021\\"
sensor = "0xbfb8"
sensor2 = "0xab52"
broadcast1 = "0x0000"
broadcast2 = "0xffff"
input = [sensor, sensor2, broadcast1, broadcast2]
output = list(itertools.combinations(input, 1)) + list(itertools.combinations(input, 2)) + list(itertools.combinations(input, 3)) + list(itertools.combinations(input, 4))
timeintervals = ["300"]

file = "cc2531_sniffer_2_3_2021_Trainingout_"

# get a list of models to evaluate
def get_models(input, x):
    df = pd.read_csv(folder + file + x + "_Seconds_Count.csv")
    X = df[list(input) + ['len_sum', 'smallest', 'largest', 'count_sum']]
    y = df['occupied']
    models = dict()
    # explore depths from 1 to 10
    for i in range(1, 10):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
        # fit the model on the whole dataset
        models[str(i)].fit(X, y)
    return models


for z in output:
    outputfile = "output_rf_" + '_'.join(z)
    f = open(folder + outputfile, "w")
    for x in timeintervals:
        # define dataset
        df = pd.read_csv(folder + file + x + "_Seconds_Count.csv")
        X = df[list(z) + ['len_sum', 'smallest', 'largest', 'count_sum']]
        y = df['occupied']
        # define the model with default hyperparameters
        model = AdaBoostClassifier(base_estimator=LogisticRegression())
        # define the grid of values to search
        grid = dict()
        grid['n_estimators'] = [10, 50, 100, 500]
        grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
        # define the evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        index = 1
        models = get_models(z, x)
        # evaluate the models and store results
        results, names = list(), list()
        print(file + x + "_Seconds_Count.csv")
        f.writelines(file + x + "_Seconds_Count.csv\n")
        for name, model in models.items():
            print("Depth: ", index)
            f.writelines("Depth: " + str(index) + "\n")
            index += 1
            # evaluate the model
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
            # execute the grid search
            grid_result = grid_search.fit(X, y)
            # summarize the best score and configuration
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            f.writelines("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
            # summarize all scores that were evaluated
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
                f.writelines("%f (%f) with: %r\n" % (mean, stdev, param))
    f.close()
f.close()