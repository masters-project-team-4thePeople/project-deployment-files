import json
import requests
import streamlit as st
from transformers import pipeline

st.title('Summarized Video Transcript Classification using Few Shot Learning and Zero Shot Learning')

st.header('Enter Youtube Video URL')
video_url = st.text_input('YouTube Video', '')
st.caption('Processing URL :   ' + str(video_url))

if video_url:
    st.header('Transcription of Video')

    # get transcript results
    url = "http://127.0.0.1:8001/"
    payload = {'youtube_url': video_url}
    files = []
    headers = {}

    transcription_response = requests.request("POST", url, headers=headers, data=payload, files=files)
    transcription_text = json.loads(transcription_response.text)['video_text']

    st.caption(transcription_text)

    if transcription_text:
        st.header('Summarized Text')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Zero Shot Text Summarizer")
            url = "http://127.0.0.1:8002/zero_shot_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            zero_short_summarized = json.loads(response.text)['results'][0]['summary_text']
            st.caption(zero_short_summarized)

        with col2:
            st.subheader("Bert Text Summarizer")
            url = "http://127.0.0.1:8002/bert_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            bert_summarized = json.loads(response.text)['results']
            st.caption(bert_summarized)

        with col3:
            st.subheader("GPT Text Summarizer")
            url = "http://127.0.0.1:8002/gpt_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            gpt_summarized = json.loads(response.text)['results']
            st.caption(gpt_summarized)

        with col4:
            st.subheader("XLNET Text Summarizer")
            url = "http://127.0.0.1:8002/xlnet_summarizer/"
            payload = {'video_text': transcription_text}
            files = []
            headers = {}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            xlnet_summarized = json.loads(response.text)['results']
            st.caption(xlnet_summarized)

        st.caption('Processing Video Category Classification')

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

        st.subheader("Video Category")
        st.subheader(", ".join(category_list))

        if category_list:
            st.subheader("More Information About Results")

            st.text("Zero Shot Results")
            st.code(zero_short_result)

            st.text("BERT Results")
            st.code(bert_result)

            st.text("GPT Results")
            st.code(gpt_result)

            st.text("XLNET Results")
            st.code(xlnet_result)
