from django.shortcuts import render
from .models import Networkif
from background_task.models import Task
from django.http import JsonResponse
import os
import json


from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

timestamp = {}
# Create your views here.
@csrf_exempt
def index(request):
    if request.method == 'GET':
        title = 'title'
        obj = Networkif.objects.first()
        title_value = getattr(obj, title)

        sensorReadings = 'sensorReadings'
        obj = Networkif.objects.first()
        sensorReadings_value = getattr(obj, sensorReadings)

        context = {
            'title': title_value,
            'sensorReadings': sensorReadings_value,
        }
        dirname = os.path.dirname(os.path.dirname(__file__))
        folder = os.path.join(dirname, 'networkif/templates/')
        return render(request, folder + '/base.html', context=context)
    elif request.method == 'POST':
        title = 'title'
        obj = Networkif.objects.first()
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
            obj.sensorData.append(obj.timestamp)
            obj.save()

        resp_data = {
            'title': title_value,
            'sensorReadings': sensorData
        }
        print(request.POST)
        return JsonResponse(resp_data, status=200)

def networkif(request):
    if request.is_ajax() and request.method == 'GET':
        title = 'title'
        obj = Networkif.objects.first()
        title_value = getattr(obj, title)

        sensorReadings = 'sensorReadings'
        obj = Networkif.objects.first()
        sensorReadings_value = getattr(obj, sensorReadings)

        resp_data = {
            'title': title_value,
            'sensorReadings': sensorReadings_value
        }

    return JsonResponse(resp_data, status=200)

