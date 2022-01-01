import sys

from django.shortcuts import render
from django.http import HttpResponse
from django.utils import translation
from background_task.models import Task

from .changepoint import run_changepoint_5, readSensor, return_graph
from .models import ChangePoint

def index(request):
    if not Task.objects.filter(verbose_name="changepoint").exists():
        run_changepoint_5(repeat=10, verbose_name="changepoint")
    readSensor()

    title = 'title'
    obj = ChangePoint.objects.first()
    title_value = getattr(obj, title)

    changepoint = 'changepoint'
    obj = ChangePoint.objects.first()
    changepoint_value = getattr(obj, changepoint)
    context = {
        'title': title_value,
        'changepoint': changepoint_value,
    }
    return render(request, 'base.html', context=context)
