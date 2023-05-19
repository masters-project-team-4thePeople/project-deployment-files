from __future__ import unicode_literals
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import pipeline
import requests


# API View here
class ClassificationAPI(APIView):
    def get(self, request):
        return Response({"status": "success",
                         "data": "Success Request"},
                        status=status.HTTP_200_OK)

    def post(self, request):
        try:
            # get youtube video link
            youtube_video_link = request.data['youtube_video']

            # get transcript results
            url = "http://127.0.0.1:8001/"
            payload = {'youtube_url': youtube_video_link}
            files = []
            headers = {}

            transcription_response = requests.request("POST", url, headers=headers, data=payload, files=files)
            transcription_text = json.loads(transcription_response.text)['video_text']

            # get summarized video results from ZeroShot Summarizer
            url = "http://127.0.0.1:8002/zero_shot_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            zero_short_summarized = json.loads(response.text)['results'][0]['summary_text']

            # get summarized video results from BERT Summarizer
            url = "http://127.0.0.1:8002/bert_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            bert_summarized = json.loads(response.text)['results']

            # get summarized video results from GPT2 Summarizer
            url = "http://127.0.0.1:8002/gpt_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            gpt_summarized = json.loads(response.text)['results']

            # get summarized video results from XLNET Summarizer
            url = "http://127.0.0.1:8002/xlnet_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            xlnet_summarized = json.loads(response.text)['results']

            # Zero Shot Classification
            classifier = pipeline("zero-shot-classification",
                                  model="facebook/bart-large-mnli")
            all_categories = [
                "Film & Animation",
                "Autos & Vehicles",
                "Music",
                "Pets & Animals",
                "Sports",
                "Short Movies",
                "Travel & Events",
                "Gaming",
                "Videoblogging",
                "People & Blogs",
                "Comedy",
                "Entertainment",
                "News & Politics",
                "How to & Style",
                "Education",
                "Science & Technology",
                "Nonprofits & Activism",
                "Movies",
                "Anime/Animation",
                "Action/Adventure",
                "Classics",
                "Comedy",
                "Documentary",
                "Drama",
                "Family",
                "Foreign",
                "Horror",
                "Sci-Fi/Fantasy",
                "Thriller",
                "Shorts",
                "Shows",
                "Trailers"
            ]
            print("*" * 10)
            print(zero_short_summarized)
            print("*" * 10)
            print(bert_summarized)
            print("*" * 10)
            print(gpt_summarized)
            print("*" * 10)
            print(xlnet_summarized)
            print("*" * 10)

            zero_short_response = classifier(zero_short_summarized, all_categories, multi_label=True)
            bert_response = classifier(bert_summarized, all_categories, multi_label=True)
            gpt_response = classifier(gpt_summarized, all_categories, multi_label=True)
            xlnet_response = classifier(xlnet_summarized, all_categories, multi_label=True)

            zero_short_result = {
                zero_short_response['labels'][0]: zero_short_response['scores'][0],
                zero_short_response['labels'][1]: zero_short_response['scores'][1],
                zero_short_response['labels'][2]: zero_short_response['scores'][2],
            }

            bert_result = {
                bert_response['labels'][0]: bert_response['scores'][0],
                bert_response['labels'][1]: bert_response['scores'][1],
                bert_response['labels'][2]: bert_response['scores'][2],
            }

            gpt_result = {
                gpt_response['labels'][0]: gpt_response['scores'][0],
                gpt_response['labels'][1]: gpt_response['scores'][1],
                gpt_response['labels'][2]: gpt_response['scores'][2],
            }

            xlnet_result = {
                xlnet_response['labels'][0]: xlnet_response['scores'][0],
                xlnet_response['labels'][1]: xlnet_response['scores'][1],
                xlnet_response['labels'][2]: xlnet_response['scores'][2],
            }

            print(zero_short_result)
            print(bert_result)
            print(gpt_result)
            print(xlnet_result)

            category_list = []

            for cat in zero_short_result.keys():
                category_list.append(cat)

            for cat in bert_result.keys():
                category_list.append(cat)

            for cat in gpt_result.keys():
                category_list.append(cat)

            for cat in xlnet_result.keys():
                category_list.append(cat)

            category_list = list(set(category_list))

            return Response({"status": "success",
                             "zero_shot": zero_short_result,
                             "bert": bert_result,
                             "gpt": gpt_result,
                             "xlnet": xlnet_result,
                             "Category": ",".join(category_list)},
                            status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"status": "failed"},
                            status=status.HTTP_400_BAD_REQUEST)

