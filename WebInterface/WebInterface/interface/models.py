from django.db import models
# Create your models here.
from django.urls import reverse

class Interface(models.Model):
    title = models.CharField(max_length=50)
    timestamp = models.JSONField(blank=True, default=dict)
    sensorData = models.JSONField(blank=True, default=list)
    sensorCounts = models.JSONField(blank=True, default=dict)
    sensorCountsChangePoint = models.JSONField(blank=True, default=list)
    occupied_hist = models.JSONField(blank=True, default=list)
    occupied = models.FloatField()
    changepoint_hist = models.JSONField(blank=True, default=list)
    changepoint = models.FloatField()
    countsIndex = models.IntegerField()
    changePointPlot = models.JSONField(blank=True, default=dict)
    supervisedPlot = models.JSONField(blank=True, default=dict)

class Meta:
    db_table = "interface"

def __str__(self):
    return self.field_name

def get_absolute_url(self):
    """Returns the url to access a particular instance of the model."""
    return reverse('model-detail-view', args=[str(self.id)])
