from django.shortcuts import render
from django.http import HttpResponse
from background_task import background
from django.template.defaultfilters import register
import os

from .models import Update

# Create your views here.
def index(request):
    update_cal()
    title = 'title'
    obj = Update.objects.first()
    title_value = getattr(obj, title)

    cal = 'cal'
    obj = Update.objects.first()
    cal_value = getattr(obj, cal)
    context = {
        'title': title_value,
        'cal': cal_value,
    }

    dirname = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(dirname, 'update/templates/')
    return render(request, folder + 'base.html', context=context)

@background(schedule=15)
def update_cal():
    add_timezone('test')

def add_timezone(value):
    adjusted_tz = ...
    return adjusted_tz

register.filter('add_timezone', add_timezone)