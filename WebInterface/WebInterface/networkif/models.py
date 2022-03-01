from django.db import models
# Create your models here.
from django.urls import reverse
from django.contrib.postgres.fields import ArrayField

def get_default():
    return ['tutorial', 'django']

class Networkif(models.Model):
    title = models.CharField(max_length=50)
    timestamp = models.JSONField(blank=True, default=dict)
    sensorData = models.JSONField(blank=True, default=dict)

class Meta:
    db_table = "networkif"

def __str__(self):
    return self.field_name

def get_absolute_url(self):
    """Returns the url to access a particular instance of the model."""
    return reverse('model-detail-view', args=[str(self.id)])
