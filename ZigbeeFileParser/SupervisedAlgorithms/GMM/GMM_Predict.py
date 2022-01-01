import pandas as pd
from sklearn.mixture import GaussianMixture
import joblib
sensor = '0xbfb8'
file = "3_23_2021out_5_Seconds_Count_Data_Only_Parse_Entrance_3_State.csv"
df = pd.read_csv("C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\" + file)
X = df[[sensor, 'len_sum', 'smallest', 'largest', 'count_sum']]
y = df['occupied']
n_components = 3
max_iter = 300
covariance_type = 'tied'
model = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=covariance_type)
# fit the model on the whole dataset
model.fit(X, y)

test = X.values[0]
print(test)
result = model.predict(test.reshape(1, -1))
print(result)
