import random
import numpy as np
import json
from django.views import View
from django.shortcuts import render, HttpResponse
from plugins.duia.httpreturn import http_return as hr
from sentiment_analysis.db import SentimentAnalysisDict
from sentiment_analysis.core.get_result import test as df
from sentiment_analysis.core.get_result import *
# Create your views here.


class SentimentAnalysis(View):
    def make_data(self, class_B):
        color_list = [
            '#d56d68', '#d897a8', '#dabe88', '#d47868', '#da0d68',
            '#975e6d', '#ef2d36', '#aeb92c', '#ebb40f', '#8f1c53',
            '#e0719c', '#c94a44', '#4eb849', '#e1c315', '#ba9232',
            '#f99e1c', '#b53b54', '#f68a5c', '#9ea718', '#b34039',
            '#ef5a78', '#a5446f', '#baa635', '#94a76f', '#fefef4',
            '#da1d23', '#dd4c51', '#f7a128', '#d0b24f', '#8b8c90',
            '#dd4c51', '#f2684b', '#f26355', '#8eb646', '#beb276',
            '#3e0317', '#e73451', '#e2631e', '#faef07', '#9db2b7',
            '#e62969', '#e65656', '#7eb138', '#b09733', '#187a2f',
            '#6569b0', '#f89a1c', '#fde404', '#c1ba07', '#0aa3b5',
        ]
        class_A = {
            'location': round(np.mean([class_B[class_b] for class_b in class_B if class_b.startswith('location')])),
            'service': round(np.mean([class_B[class_b] for class_b in class_B if class_b.startswith('service')])),
            'price': round(np.mean([class_B[class_b] for class_b in class_B if class_b.startswith('price')])),
            'environment': round(
                np.mean([class_B[class_b] for class_b in class_B if class_b.startswith('environment')])),
            'dish': round(np.mean([class_B[class_b] for class_b in class_B if class_b.startswith('dish')])),
            'others': round(np.mean([class_B[class_b] for class_b in class_B if class_b.startswith('others')])),
        }
        echart_data = [{
            'name': SentimentAnalysisDict[class_a],
            'itemStyle': {'color': color_list.pop()},
            'children': [{
                'name': SentimentAnalysisDict[class_b],
                'itemStyle': {'color': color_list.pop()},
                'children': [{
                    'name': class_B[class_b],
                    'itemStyle': {'color': color_list.pop()},
                    'value': class_B[class_b] + 3,
                }]
            } for class_b in class_B if class_b.startswith(class_a)],
        } for class_a in class_A]
        return class_A, echart_data

    def get(self, request):
        return render(request, 'sentiment_analysis/sentiment_analysis.html')

    def post(self, request):
        try:
            contents = request.POST.get('contents')
            if not contents: raise Exception
            class_B = get_result(contents)
            class_A, echart_data = self.make_data(class_B)
            ret = hr(status=True, message={'echart_data': echart_data, 'class_A': class_A}, code=200)

        except Exception as e:
            ret = hr(status=False, message="请输入文本...", code=201)

        finally:
            return HttpResponse(json.dumps(ret))

    def put(self, request):
        random_df = df.ix[random.randrange(15000)]
        content = random_df['content']
        pred = json.loads(random_df['pred'])
        y_cols = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
                  'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience',
                  'service_serving_speed', 'price_level', 'price_cost_effective', 'price_discount',
                  'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
                  'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation', 'others_overall_experience',
                  'others_willing_to_consume_again']
        class_B = dict(zip(y_cols, pred))
        class_A, echart_data = self.make_data(class_B)
        
        ret = hr(status=True, message={'echart_data': echart_data, 'class_A': class_A, 'content': content}, code=200)
        return HttpResponse(json.dumps(ret))
