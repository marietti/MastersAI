from django.shortcuts import render
from .models import Supervised
from background_task.models import Task

from django.http import JsonResponse
from .supervised import readSensor, load_value
import os
import numpy as np
import collections
import json

import matplotlib.pyplot as plt
from io import StringIO

x = np.arange(0, 100, 1)
d = collections.deque(maxlen=100)

# Create your views here.
def index(request):
    if not Task.objects.filter(verbose_name="supervised").exists():
        load_value(repeat=10, verbose_name="supervised")
    readSensor()

    title = 'title'
    obj = Supervised.objects.first()
    title_value = getattr(obj, title)

    occupied = 'occupied'
    obj = Supervised.objects.first()
    occupied_value = getattr(obj, occupied)
    context = {
        'title': title_value,
        'occupied': occupied_value,
    }
    dirname = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(dirname, 'supervised/templates/')
    return render(request, folder + '/base.html', context=context)

def supervised(request):
    if request.is_ajax() and request.method == 'GET':
        title = 'title'
        obj = Supervised.objects.first()
        title_value = getattr(obj, title)

        occupied = 'occupied'
        obj = Supervised.objects.first()
        occupied_value = getattr(obj, occupied)

        changePointPlot = 'changePointPlot'
        obj = Supervised.objects.first()
        changepointplot_value = getattr(obj, changePointPlot)

        if len(d) < 100:
            [i for i in range(100) if d.append(0.0)]
        d.append(float(occupied_value))
        obj.changePointPlot = json.dumps(list(d))
        obj.save()
        resp_data = {
            'title': title_value,
            'occupied': occupied_value,
            'graph': return_graph(changepointplot_value, x),
        }

    return JsonResponse(resp_data, status=200)


def return_graph(d, x):
    plt.ioff()
    fig = plt.figure()

    plt.plot(x, json.loads(d))

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    plt.close()
    data = imgdata.getvalue()
    return data
