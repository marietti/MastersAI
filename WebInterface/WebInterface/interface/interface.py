import joblib
import os
import sklearn
import random
import pandas as pd
import joblib
import numpy as np
import json

from background_task import background
from .models import Interface
from .changepoint_calc import change_detection_single

loaded_model = sklearn.base

def readModel():
    global loaded_model
    folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, os.pardir, 'ZigbeeFileParser/SupervisedAlgorithms/GMM/'))
    filename = folder + '/finalized_model_GMM.sav'
    loaded_model = joblib.load(filename)


@background(schedule=5)
def load_value():
    global loaded_model

    if isinstance(loaded_model, type(sklearn.base)):
        readModel()
    id = Interface.objects.first()
    obj = Interface.objects.get(id=id.id)
    for entry in obj.sensorData:
        if '0x1c88' == entry['DstPANID']:

            if 'len_sum' in obj.sensorCounts:
                obj.sensorCounts['len_sum'] += len(entry['RawData'])
            else:
                obj.sensorCounts['len_sum'] = len(entry['RawData'])

            if 'smallest' in obj.sensorCounts:
                if len(entry['RawData']) < obj.sensorCounts['smallest']:
                    obj.sensorCounts['smallest'] = len(entry['RawData'])
            else:
                obj.sensorCounts['smallest'] = len(entry['RawData'])

            if 'largest' in obj.sensorCounts:
                if len(entry['RawData']) > obj.sensorCounts['largest']:
                    obj.sensorCounts['largest'] = len(entry['RawData'])
            else:
                obj.sensorCounts['largest'] = len(entry['RawData'])

            if 'count_sum' in obj.sensorCounts:
                obj.sensorCounts['count_sum'] += 1
            else:
                obj.sensorCounts['count_sum'] = 1

            if 'SrcAddr' in obj.sensorCounts:
                obj.sensorCounts[entry['SrcAddr']] = obj.sensorCounts['SrcAddr'] + 1
            else:
                obj.sensorCounts[entry['SrcAddr']] = 1

            if 'DstAddr' in obj.sensorCounts:
                obj.sensorCounts[entry['DstAddr']] = obj.sensorCounts['SrcAddr'] + 1
            else:
                obj.sensorCounts[entry['SrcAddr']] = 1
            obj.save(update_fields=["sensorCounts"])
    n = 50
    k = 10
    if len(obj.sensorData) > 0:
        if len(obj.sensorCountsChangePoint) < n * 2 + k - 1:
            obj.sensorCountsChangePoint.append(obj.sensorCounts)
            obj.save(update_fields=["sensorCountsChangePoint"])
        else:
            obj.changepoint = float(calc_changepoint())
            obj.save(update_fields=["changepoint"])
            if len(obj.changepoint_hist):
                obj.changepoint_hist = list()
            obj.changepoint_hist.append(obj.changepoint)
            obj.save(update_fields=["changepoint_hist"])
            obj.sensorCountsChangePoint.append(obj.sensorCounts)
            obj.sensorCountsChangePoint.pop(0)
            obj.save(update_fields=["sensorCountsChangePoint"])
    if len(obj.sensorData) > 0:
        obj.occupied = calc_supervised()
        obj.save(update_fields=["occupied"])
        if len(obj.occupied_hist):
            obj.occupied_hist = list()
        obj.occupied_hist.append(obj.occupied)
        obj.save(update_fields=["occupied_hist"])
    obj.sensorData = list()
    obj.save(update_fields=["sensorData"])

def calc_changepoint():
    id = Interface.objects.first()
    obj = Interface.objects.get(id=id.id)
    df = pd.DataFrame(obj.sensorCountsChangePoint)
    if not df.empty:
        y = np.reshape(np.array(df["0xbfb8"].tolist()), (1, np.array(df["0xbfb8"].tolist()).size))
        n = 50
        k = 10
        if y.size >= n*2+k-1:
            random.seed(1)
            noise = np.random.normal(0, 1, y.size)
            y = y + noise
            alpha = .01
            index_y = n*2+k-1
            y_read = y[:, 0:index_y]
            [s, _, _] = change_detection_single(y_read, n, k, alpha, 5)
            print(float(s))
            return float(s)
    else:
        return 0.0
def calc_supervised():
    id = Interface.objects.first()
    obj = Interface.objects.get(id=id.id)
    df = pd.DataFrame.from_dict([{k: obj.sensorCounts[k] for k in ('0xbfb8', 'len_sum', 'smallest', 'largest', 'count_sum') if k in obj.sensorCounts}])
    if not df.empty:
        test = df.values[0]
        print(loaded_model.predict(test.reshape(1, -1)))
        return int(loaded_model.predict(test.reshape(1, -1))[0])
    return -1
