# Generated by Django 3.1.6 on 2021-02-25 21:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('changepoint', '0005_changepoint_sensorcounts'),
    ]

    operations = [
        migrations.AddField(
            model_name='changepoint',
            name='countsIndex',
            field=models.IntegerField(default=0, verbose_name=0),
            preserve_default=False,
        ),
    ]
