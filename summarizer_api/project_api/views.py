from __future__ import unicode_literals
import json
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import pipeline
from summarizer import Summarizer, TransformerSummarizer


# API View here
class ZeroShortSummarizerAPI(APIView):
    def get(self, request):
        return Response({"status": "success",
                         "data": "Success Request"},
                        status=status.HTTP_200_OK)

    def post(self, request):
        try:
            video_text = request.data['video_text']
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            response = summarizer(video_text,
                                  do_sample=False)

            return Response({"status": "success",
                             "results": response},
                            status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            return Response({"status": "failed"},
                            status=status.HTTP_400_BAD_REQUEST)


class BertSummarizerAPI(APIView):
    def get(self, request):
        return Response({"status": "success",
                         "data": "Success Request"},
                        status=status.HTTP_200_OK)

    def post(self, request):
        try:
            video_text = request.data['video_text']
            bert_model = Summarizer()
            bert_summary = ''.join(bert_model(video_text))

            return Response({"status": "success",
                             "results": bert_summary},
                            status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            return Response({"status": "failed"},
                            status=status.HTTP_400_BAD_REQUEST)


class GPTSummarizerAPI(APIView):
    def get(self, request):
        return Response({"status": "success",
                         "data": "Success Request"},
                        status=status.HTTP_200_OK)

    def post(self, request):
        try:
            video_text = request.data['video_text']
            GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
            GPT2_summary = ''.join(GPT2_model(video_text))

            return Response({"status": "success",
                             "results": GPT2_summary},
                            status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            return Response({"status": "failed"},
                            status=status.HTTP_400_BAD_REQUEST)


class XLNETSummarizerAPI(APIView):
    def get(self, request):
        return Response({"status": "success",
                         "data": "Success Request"},
                        status=status.HTTP_200_OK)

    def post(self, request):
        try:
            video_text = request.data['video_text']
            xlnet_model = TransformerSummarizer(transformer_type="XLNet",
                                                transformer_model_key="xlnet-base-cased")
            xlnet_summary = ''.join(xlnet_model(video_text))

            return Response({"status": "success",
                             "results": xlnet_summary},
                            status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            return Response({"status": "failed"},
                            status=status.HTTP_400_BAD_REQUEST)
