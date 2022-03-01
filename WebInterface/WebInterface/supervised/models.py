from django.db import models
# Create your models here.
from django.urls import reverse

class Supervised(models.Model):
    title = models.CharField(max_length=50)
    occupied = models.FloatField()
    sensorCounts = models.JSONField(blank=True, default=dict)
    countsIndex = models.IntegerField()
    changePointPlot = models.JSONField(blank=True, default=dict)

class Meta:
    db_table = "supervised"

def __str__(self):
    return self.field_name

def get_absolute_url(self):
    """Returns the url to access a particular instance of the model."""
    return reverse('model-detail-view', args=[str(self.id)])
