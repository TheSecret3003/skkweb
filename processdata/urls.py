from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recovery_factor.html', views.results, name='results'),
    path('profile', views.profile, name='profile')
]
