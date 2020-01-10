"""day5 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from automatic_summary import views as automatic_summary_views
from sentiment_analysis import views as sentiment_analysis_views

urlpatterns = [
    url(r'^$', automatic_summary_views.Index.as_view(), name='index'),
    url(r'^text_summary/$', automatic_summary_views.TextSummary.as_view(), name='text_summary'),
    url(r'^sentiment_analysis/$', sentiment_analysis_views.SentimentAnalysis.as_view(), name='sentiment_analysis'),
]
