# Generated by Django 3.1.6 on 2021-02-22 00:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('changepoint', '0003_changepoint_sensorcounts'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='changepoint',
            name='sensorCounts',
        ),
    ]
