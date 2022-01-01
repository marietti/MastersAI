import pandas as pd
import numpy
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt\

sensor = "0xbfb8"
sensor2 = "0xab52"
broadcast1 = "0x0000"
broadcast2 = "0xffff"
file_3 = "cc2531_sniffer_2_3_2021out_5_Seconds_Count_Data_Only_Parse_Entrance_3_State.csv"
file = "cc2531_sniffer_2_3_2021out_5_Seconds_Count_Data_Only_Parse_Entrance_3_State_trimmed_more.csv"
folder_3 = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\3_23_2021\\"
folder = "C:\\Users\\nickb\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Training\\ChangePoint\\2_3_2021\\"

df = pd.read_csv(folder + file)
df_3 = pd.read_csv(folder + file_3)
X = df[[sensor, broadcast1, broadcast2, 'len_sum', 'smallest', 'largest', 'count_sum']]
X_3 = df_3[[sensor, broadcast1, broadcast2, 'len_sum', 'smallest', 'largest', 'count_sum']]
y = df['occupied']
y_3 = df_3['occupied']

n_components = 3
max_iter = 300
covariance_type = 'tied'
model = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=covariance_type)

# fit the model on the whole dataset
model.fit(X_3, y_3)
fig = plt.figure()
build_array = lambda test: model.predict(test.reshape(1, -1))

plt.plot([element * max(df['occupied']) for element in df['occupied']], label="Occupied")
array = [build_array(x) for x in X.values]
plt.plot([element * max(array) for element in array], label="Predicated")
plt.legend()
plt.show()
