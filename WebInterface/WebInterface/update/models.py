from django.db import models
from django.contrib import admin
# Create your models here.
from django.urls import reverse


class Update(models.Model):
    title = models.CharField(max_length=50)
    cal = models.FloatField()

class Meta:
    db_table = "update"

def __str__(self):
    return self.field_name

def get_absolute_url(self):
    """Returns the url to access a particular instance of the model."""
    return reverse('model-detail-view', args=[str(self.id)])
