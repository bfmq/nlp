import random
import json
import requests
from django.views import View
from django.shortcuts import render, HttpResponse
# from automatic_summary.core.get_sentences import SifEmbedding, df
from plugins.duia.httpreturn import http_return as hr

# Create your views here.


class Index(View):
    def get(self, request):
        try:
            city_url = 'http://ip.taobao.com/service/getIpInfo.php?ip='
            remote_addr = request.META.get('REMOTE_ADDR')
            city_response = requests.get(city_url+remote_addr).json()
            country, region = city_response['data']['country'], city_response['data']['region']

        except Exception as e:
            country, region = '中国', '北京'

        finally:
            return render(request, 'layout/layout_promo.html', locals())


class TextSummary(View):
    def get(self, request):
        return render(request, 'automatic_summary/text_summary.html')

    def post(self, request):
        try:
            contents = request.POST.get('contents')
            title = request.POST.get('title')
            if not contents:
                raise Exception
            sif_obj = SifEmbedding(contents, title)
            r = sif_obj.get_summarization()
            ret = hr(status=True, message=r, code=200)

        except Exception as e:
            ret = hr(status=False, message="请输入文本...", code=201)

        finally:
            return HttpResponse(json.dumps(ret))

    def put(self, request):
        x = random.randrange(499)
        data = {
            'content': df['content'][x],
            'title': df['title'][x],
        }
        ret = hr(status=True, message=data, code=200)
        return HttpResponse(json.dumps(ret))
