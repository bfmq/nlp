import random
import json
from collections import defaultdict
from django.views import View
from django.shortcuts import render, HttpResponse
from automatic_summary.core.get_sentences import df
from plugins.duia.httpreturn import http_return as hr
from automatic_get.core.dependency_parser import get_speech
# Create your views here.


class AutomaticGet(View):
    def get(self, request):
        return render(request, 'automatic_get/automatic_get.html')

    def post(self, request):
        try:
            contents = request.POST.get('contents').strip()
            all_list = get_speech(contents)

            # 将同名合并，言论合并
            all_dict = defaultdict(list)
            for alls in all_list:
                key = alls[0]
                value = alls[1:]
                for k in all_dict:
                    if set(key).issubset(set(k)) or set(k).issubset(set(key)):
                        all_dict[k].append(value)
                        break
                else:
                    all_dict[key].append(value)

            ret = {
                'name': '原句',
                'children': [
                    {
                        'name': who,
                        'children': [{
                            'name': x[0],
                            'children': [{
                                'name': x[1],
                                # 'value': x[2]
                            }]
                        } for x in all_dict[who]]
                    } for who in all_dict
                ]
            }

        except Exception as e:
            ret = {'name': '原句'}

        finally:
            return HttpResponse(json.dumps(ret))

    def put(self, request):
        x = random.randrange(499)
        data = {
            'content': df['content'][x],
        }
        ret = hr(status=True, message=data, code=200)
        return HttpResponse(json.dumps(ret))

