# Generated by Django 3.1.6 on 2021-02-26 03:09

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('changepoint', '0009_auto_20210225_1826'),
    ]

    operations = [
        migrations.AlterField(
            model_name='changepoint',
            name='sensorCounts',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), size=None), default=0, size=None),
            preserve_default=False,
        ),
    ]