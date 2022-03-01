# Generated by Django 3.2.6 on 2021-08-30 02:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Networkif',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=50)),
                ('timestamp', models.JSONField(null=True)),
                ('sensorReadings', models.JSONField(null=True)),
            ],
        ),
    ]