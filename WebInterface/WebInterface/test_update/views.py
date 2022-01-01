from django.shortcuts import render
from django.http import HttpResponse
from background_task import background
from django.template.defaultfilters import register
from django.http import JsonResponse

from background_task.models import Task
from .update_background import run_changepoint_5
from .models import Update_Test
import os

# Create your views here.
def index(request):
    if not Task.objects.filter(verbose_name="changepoint_test").exists():
        run_changepoint_5(repeat=10, verbose_name="changepoint_test")

    title = 'title'
    obj = Update_Test.objects.first()
    title_value = getattr(obj, title)

    changepoint = 'changepoint'
    obj = Update_Test.objects.first()
    changepoint_value = getattr(obj, changepoint)
    context = {
        'title': title_value,
        'changepoint': changepoint_value,
    }

    dirname = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(dirname, 'test_update/templates/')
    return render(request, folder + 'base.html', context=context)

def update(request):

    if request.is_ajax() and request.method == 'GET':
        title = 'title'
        obj = Update_Test.objects.first()
        title_value = getattr(obj, title)

        changepoint = 'changepoint'
        obj = Update_Test.objects.first()
        changepoint_value = getattr(obj, changepoint)

        resp_data = {
            'title': title_value,
            'changepoint': changepoint_value,
        }

    return JsonResponse(resp_data, status=200)
