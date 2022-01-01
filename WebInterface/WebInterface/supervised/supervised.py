import joblib
import os

import sklearn
from background_task import background
import pandas as pd
import joblib
from .models import Supervised

loaded_model = sklearn.base

def readModel():
    global loaded_model
    folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, os.pardir, 'ZigbeeFileParser/SupervisedAlgorithms/GMM/'))
    filename = folder + '/finalized_model_GMM.sav'
    loaded_model = joblib.load(filename)


def readSensor():
    file = "/3_23_2021out_5_Seconds_Count_Data_Only_Parse_Entrance_3_State.csv"
    folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, os.pardir, 'ZigbeeFileParser/FileParsing/Training/ChangePoint/3_23_2021/'))

    df = pd.read_csv(folder + file)
    obj = Supervised.objects.first()
    input = df[['0xbfb8', 'len_sum', 'smallest', 'largest', 'count_sum']]
    obj.sensorCounts = input.to_json()
    obj.CountsIndex = 0
    obj.save()


@background(schedule=5)
def load_value():
    global loaded_model

    if isinstance(loaded_model, type(sklearn.base)):
        readModel()
    obj = Supervised.objects.first()
    df = pd.read_json(obj.sensorCounts)
    index_offset = obj.countsIndex
    if index_offset >= df.values.size:
        index_offset = 0
    test = df.values[index_offset]
    obj.countsIndex = index_offset + 1
    obj.occupied = loaded_model.predict(test.reshape(1, -1))
    obj.save()
