from django.shortcuts import render
from .models import Interface
from background_task.models import Task
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .interface import load_value
import os
import numpy as np
import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from io import StringIO

x = np.arange(0, 100, 1)
changepoint_d = collections.deque(maxlen=100)
supervised_d = collections.deque(maxlen=100)

# Create your views here.
@csrf_exempt
def index(request):
    if request.method == 'GET':
        if not Task.objects.filter(verbose_name="interface").exists():
            load_value(repeat=10, verbose_name="interface")

        title = 'title'
        id = Interface.objects.first()
        obj = Interface.objects.get(id=id.id)
        title_value = getattr(obj, title)

        occupied = 'occupied'
        id = Interface.objects.first()
        obj = Interface.objects.get(id=id.id)
        occupied_value = getattr(obj, occupied)
        context = {
            'title': title_value,
            'occupied': occupied_value,
        }
        dirname = os.path.dirname(os.path.dirname(__file__))
        folder = os.path.join(dirname, 'interface/templates/')
        return render(request, folder + '/base.html', context=context)
    elif request.method == 'POST':
        title = 'title'
        id = Interface.objects.first()
        obj = Interface.objects.get(id=id.id)
        title_value = getattr(obj, title)
        sensorData = 'sensorData'
        sensorData_value = getattr(obj, sensorData)
        timestamp = 'timestamp'
        timestamp_value = getattr(obj, timestamp)
        data = request.POST
        if data.__contains__('TimeStamp'):
            obj.timestamp = data.dict()
            obj.save()
        if data.__contains__('Res'):
            obj.timestamp.update(data.dict())
            if isinstance(obj.sensorData, dict):
                obj.sensorData = list()
            obj.sensorData.append(obj.timestamp)
            obj.save()

        resp_data = {
            'title': title_value,
            'sensorReadings': sensorData
        }
        print(request.POST)
        return JsonResponse(resp_data, status=200)

def interface(request):
    if request.is_ajax() and request.method == 'GET':
        id = Interface.objects.first()
        obj = Interface.objects.get(id=id.id)

        title = 'title'
        occupied = 'occupied'
        changepoint = 'changepoint'

        title_value = getattr(obj, title)
        occupied_value = getattr(obj, occupied)
        changepoint_value = getattr(obj, changepoint)

        if len(changepoint_d) < 100:
            [i for i in range(100) if changepoint_d.append(0.0)]
        changepoint_d.append(float(changepoint_value))
        obj.changePointPlot = json.dumps(list(changepoint_d))

        if len(supervised_d) < 100:
            [i for i in range(100) if supervised_d.append(0.0)]
        supervised_d.append(float(occupied_value))
        obj.supervisedPlot = json.dumps(list(supervised_d))

        imgdata = StringIO()
        imgdata2 = StringIO()
        plt.plot(x, json.loads(obj.changePointPlot))
        plt.savefig(imgdata, format='svg')
        changepoint_plot = return_graph(imgdata)
        plt.close('all')
        plt2.plot(x, json.loads(obj.supervisedPlot))
        plt2.savefig(imgdata2, format='svg')
        supervised_plot = return_graph(imgdata2)
        # Closes all the figure windows.
        plt2.close('all')
        resp_data = {
            'title': title_value,
            'occupied': occupied_value,
            'changepoint': changepoint_value,
            'graph': changepoint_plot,
            'graph2': supervised_plot,
        }

        obj.save()
    return JsonResponse(resp_data, status=200)


def return_graph(imgdata):
    imgdata.seek(0)
    data = imgdata.getvalue()
    return data
