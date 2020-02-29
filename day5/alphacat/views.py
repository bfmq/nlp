import random
import json
import requests
from django.views import View
from django.shortcuts import render, HttpResponse
from plugins.duia.httpreturn import http_return as hr
from alphacat.core.spider_man import spider_man
from alphacat.core.syntax_tree import get_generation_by_gram
from alphacat.core.dialogue_robot import get_answer
# Create your views here.


class Cat(View):
    def get(self, request):
        return render(request, 'alphacat/alphacat.html')

    def post(self, request):
        try:
            contents = request.POST.get('contents').strip()
            message = get_answer(contents) or spider_man(contents) or get_generation_by_gram()

        except Exception as e:
            print(e)
            message = get_generation_by_gram()

        finally:
            ret = hr(status=True, message=message, code=200)
            return HttpResponse(json.dumps(ret))
