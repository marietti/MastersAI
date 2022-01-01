# example of grid searching key hyperparameters for adaboost on a classification dataset
import itertools

from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

folder_3 = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\"
folder = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\2_3_2021\\"
sensor = "0xbfb8"
sensor2 = "0xab52"
broadcast1 = "0x0000"
broadcast2 = "0xffff"
input = [sensor, sensor2, broadcast1, broadcast2]
#output = list(itertools.combinations(input, 1)) + list(itertools.combinations(input, 2)) + list(
#    itertools.combinations(input, 3)) + list(itertools.combinations(input, 4))
output = [[sensor, sensor2, broadcast1, broadcast2],[sensor, broadcast1, broadcast2], [sensor2, broadcast1, broadcast2], [sensor2], [sensor]]
timeintervals = ["5", "10", "45", "60", "90", "120", "180", "240", "300"]
n_estimators = [10, 50, 100, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 1.0]

file = "cc2531_sniffer_2_3_2021out_"
file_3 = "3_23_2021out_"
models = list()


# get a list of models to evaluate
def get_models():
    # explore depths from 1 to 10
    for i in range(1, 10):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        for estimator in n_estimators:
            for lr in learning_rate:
                models.append(tuple((AdaBoostClassifier(base_estimator=base, n_estimators=estimator, learning_rate=lr),
                                     i, estimator, lr)))
    return models

fo = open("Custom_Grid_Search_Out_Cross_Validation_Data_Only_Parse_Entrance_3_State.csv", 'w', newline='', encoding='utf-8')
fo.writelines(
    'n_components,max_iter,covariance_type,sensors_input,true positive rate c1,true positive rate c2,true positive rate c3,true negative rate c1,'
    'true negative rate c2,true negative rate c3,positive predictive value c1,positive predictive value c2,positive predictive value c3,Negative predictive value c1,'
    'Negative predictive value c2,Negative predictive value c3,false positive rate c1,false positive rate c2,false positive rate c3,False negative rate c1,False negative '
    'rate c2,False negative rate c3,False discovery rate c1,False discovery rate c2,False discovery rate c3,Overall accuracy c1,Overall accuracy c2,Overall accuracy c3,class_majority,time_window\n')
timeintervals_count = 0
models = get_models()
for x in timeintervals:
    print('timeintervals_count ' + str(timeintervals_count) + '\n')
    timeintervals_count += 1
    # define dataset
    df = pd.read_csv(folder + file + x + "_Seconds_Count_Data_Only_Parse_Entrance_3_State.csv")
    df_3 = pd.read_csv(folder_3 + file_3 + x + "_Seconds_Count_Data_Only_Parse_Entrance_3_State.csv")
    occupied_output = df['occupied']
    ones = 0
    zeros = 0
    total = 0
    for m in occupied_output:
        total += 1
        if m == 0:
            zeros += 1
        else:
            ones += 1
    occupied_cm = zeros / total if ones / total < zeros / total else ones / total
    output_count = 0
    for z in output:
        print('output_count ' + str(output_count) + '\n')
        output_count += 1
        X = df[list(z) + ['len_sum', 'smallest', 'largest', 'count_sum']]
        X_3 = df_3[list(z) + ['len_sum', 'smallest', 'largest', 'count_sum']]
        y = df['occupied']
        y_3 = df_3['occupied']
        count = 0
        for model in models:
            print('count ' + str(count) + '\n')
            count += 1
            model[0].fit(X, y)
            y_pred = model[0].predict(X_3)
            cnf_matrix = confusion_matrix(y_3, y_pred)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            # Precision or positive predictive value
            PPV = TP / (TP + FP)
            # Negative predictive value
            NPV = TN / (TN + FN)
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            # False negative rate
            FNR = FN / (TP + FN)
            # False discovery rate
            FDR = FP / (TP + FP)
            # Overall accuracy
            ACC = (TP + TN) / (TP + FP + FN + TN)
            joined_string = ":".join(list(z))
            joined_string_TPR = ",".join([str(element) for element in list(TPR)])
            joined_string_TNR = ",".join([str(element) for element in list(TNR)])
            joined_string_PPV = ",".join([str(element) for element in list(PPV)])
            joined_string_NPV = ",".join([str(element) for element in list(NPV)])
            joined_string_FPR = ",".join([str(element) for element in list(FPR)])
            joined_string_FNR = ",".join([str(element) for element in list(FNR)])
            joined_string_FDR = ",".join([str(element) for element in list(FDR)])
            joined_string_ACC = ",".join([str(element) for element in list(ACC)])
            fo.writelines(str(model[1]) + ',' + str(model[2]) + ',' + str(model[3]) + ',' + joined_string + ',' + joined_string_TPR + ',' + joined_string_TNR + ',' + joined_string_PPV + ',' + joined_string_NPV + ',' + joined_string_FPR + ',' + joined_string_FNR + ',' + joined_string_FDR + ',' + joined_string_ACC + ',' + str(occupied_cm) + ',' + x + '\n')
fo.close()
