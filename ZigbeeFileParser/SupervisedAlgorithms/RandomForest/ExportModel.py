from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

file = "Calendar_120_Seconds_Count_3.csv"
df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\Data\\" + file)
X = df[['0xd7b2', 'len_sum', 'smallest', 'largest', 'count_sum']]
y = df['occupied']
base = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base)
# fit the model on the whole dataset
model.fit(X, y)
# Save Model Using joblib
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

result = model.score(X, y)
print(result)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X, y)
print(result)
