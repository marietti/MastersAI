import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
sensor = '0xbfb8'
file = "3_23_2021out_5_Seconds_Count_Data_Only_Parse_Entrance_3_State.csv"
df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\" + file)
X = df[[sensor, 'len_sum', 'smallest', 'largest', 'count_sum']]
y = df['occupied']
n_neighbors = 10
leaf_size = 50
weights = 'distance'
model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights)
# fit the model on the whole dataset
model.fit(X, y)
# Save Model Using joblib
# save the model to disk
filename = 'finalized_model_GMM.sav'
joblib.dump(model, filename)

result = model.score(X, y)
print(result)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X, y)
print(result)
