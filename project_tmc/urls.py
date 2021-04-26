"""project_tmc URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from base.urls import websocket
from voice_classification import views as voice_classification_views

urlpatterns = [
    path("", voice_classification_views.IndexView.as_view()),
    path("home/", voice_classification_views.home, name="home"),
    path("voice-classification/", voice_classification_views.index,
         name="voice_classification_index"),
    websocket("ws/", voice_classification_views.websocket_view),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
