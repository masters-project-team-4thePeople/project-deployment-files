from django.urls import path
from .views import ZeroShortSummarizerAPI, BertSummarizerAPI, GPTSummarizerAPI

urlpatterns = [
   path('zero_shot_summarizer/', ZeroShortSummarizerAPI.as_view(), name="zero_shot_api_views"),
   path('bert_summarizer/', BertSummarizerAPI.as_view(), name="bert_api_views"),
   path('gpt_summarizer/', GPTSummarizerAPI.as_view(), name="gpt_api_views"),
   path('xlnet_summarizer/', GPTSummarizerAPI.as_view(), name="xlnet_api_views")
]