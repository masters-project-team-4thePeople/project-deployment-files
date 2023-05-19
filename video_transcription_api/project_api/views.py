from __future__ import unicode_literals
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import youtube_dl
import whisper


# API View here
class TranscribeAPI(APIView):
    def get(self, request):
        return Response({"status": "success",
                         "data": "Success Request"},
                        status=status.HTTP_200_OK)

    def post(self, request):
        # loading options
        try:
            filename = "video.mp3"
            request_data = request.data
            video_url = request_data['youtube_url']

            ydl_opts = {
                'format': 'bestaudio/best',
                'keepvideo': False,
                'outtmpl': filename
            }

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            whisper_model = whisper.load_model("base")
            transcription = whisper_model.transcribe(filename)

            # If file exists, delete it.
            if os.path.isfile(filename):
                os.remove(filename)

            return Response({"status": "success",
                             "video_url": video_url,
                             "video_text": transcription["text"]},
                            status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"status": "failed"},
                            status=status.HTTP_400_BAD_REQUEST)


