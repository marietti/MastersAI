from django.db import models
from django.contrib import admin
# Create your models here.
from django.urls import reverse
import pandas as pd
from django.contrib.postgres.fields import ArrayField

class ChangePoint(models.Model):
    title = models.CharField(max_length=50)
    changepoint = models.FloatField()
    sensorCounts = models.JSONField(null=True)
    countsIndex = models.IntegerField()
    changePointPlot = models.JSONField(null=True)

class Meta:
    db_table = "changepoint"

def __str__(self):
    return self.field_name

def get_absolute_url(self):
    """Returns the url to access a particular instance of the model."""
    return reverse('model-detail-view', args=[str(self.id)])
