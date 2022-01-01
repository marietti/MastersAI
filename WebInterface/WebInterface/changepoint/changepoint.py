import sys

from django.http import JsonResponse
from background_task import background
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.template import loader
from django.template.defaultfilters import register

from .models import ChangePoint
from .changepoint_calc import change_detection_single
from pylab import *
import pandas as pd

import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

import collections
import json
import os

x = np.arange(0, 100, 1)
d = collections.deque(maxlen=100)

def readSensor():
    file = "/cc2531_sniffer_2_3_2021out_5_Seconds_Count.csv"
    folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, os.pardir, 'ZigbeeFileParser/FileParsing/Raw/'))

    df = pd.read_csv(folder + file)
    id = ChangePoint.objects.first()
    obj = ChangePoint.objects.get(id=id.id)
    obj.sensorCounts = df.to_json()
    obj.CountsIndex = 0
    obj.save()

@background(schedule=5)
def run_changepoint_5():
    global d
    id = ChangePoint.objects.first()
    obj = ChangePoint.objects.get(id=id.id)
    df = pd.read_json(obj.sensorCounts)
    y = np.reshape(np.array(df["0xbfb8"].tolist()), (1, np.array(df["0xbfb8"].tolist()).size))
    seed(1)
    noise = np.random.normal(0, 1, y.size)
    y = y + noise
    alpha = .01
    n = 50
    k = 10
    y = y + noise
    index_offset = obj.countsIndex
    if index_offset >= y.size - (n*2+k-1):
        index_offset = 0
    index_y = n*2+k-1 + int(index_offset)
    y_read = y[:, index_offset:index_y]
    [s, _, _] = change_detection_single(y_read, n, k, alpha, 5)
    if len(d) < 100:
        [i for i in range(100) if d.append(0.0)]
    obj.changepoint = float(s)
    d.append(float(s))
    obj.changePointPlot = json.dumps(list(d))
    obj.countsIndex = index_offset + 1
    obj.save()

def test(request):
    if request.is_ajax() and request.method == 'GET':
        # main logic here setting the value of resp_data
        title = 'title'
        obj = ChangePoint.objects.first()
        title_value = getattr(obj, title)

        changepoint = 'changepoint'
        obj = ChangePoint.objects.first()
        changepoint_value = getattr(obj, changepoint)

        changePointPlot = 'changePointPlot'
        obj = ChangePoint.objects.first()
        changepointplot_value = getattr(obj, changePointPlot)

        resp_data = {
            'title': title_value,
            'changepoint': changepoint_value,
            'graph': return_graph(changepointplot_value),
        }

    return JsonResponse(resp_data, status=200)

def return_graph(d):
    plt.ioff()
    fig = plt.figure()

    plt.plot(x, json.loads(d))

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    plt.close()
    data = imgdata.getvalue()
    return data