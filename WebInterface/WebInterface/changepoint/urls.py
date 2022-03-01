from django.urls import path

from . import views, changepoint

urlpatterns = [
    path('', views.index, name='index'),
    path(r'^$', changepoint.test, name='changepoint'),
]