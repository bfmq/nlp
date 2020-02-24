import random
import json
import requests
from django.views import View
from django.shortcuts import render, HttpResponse
from plugins.duia.httpreturn import http_return as hr

# Create your views here.


class Cat(View):
    def get(self, request):
        return render(request, 'alphacat/alphacat.html')

    def post(self, request):
        contents = request.POST.get('contents')
        print(contents)
        return HttpResponse(json.dumps('回复'))
