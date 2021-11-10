from os import name
from django.urls import path
from . import views as main_views
from django.views.decorators.csrf import csrf_exempt
from sentiment_analyser import views

urlpatterns = [
    path('base/', csrf_exempt(main_views.base), name="base"),
    #path('scoreJ/', csrf_exempt(main_views.scoreJson), name='scoreJson'),
    path('scoref/', csrf_exempt(main_views.scoreFile), name='scoreFile'),
    path('sentiment/', views.sentiment_main, name= 'sentiment_main'),
    path('analyse/', csrf_exempt(views.prediction), name= 'prediction')
]