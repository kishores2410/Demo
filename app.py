import pandas as pd
import streamlit as st
from pytube import YouTube
import openai

# Initialize OpenAI
openai.api_key = "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"

# Streamlit app
st.markdown('<h1>YouTube GPT 🤖</h1>', unsafe_allow_html=True)
st.write("Start a chat with any YouTube video. Paste the YouTube link below and ask a question:")

# User input
youtube_link = st.text_input("Paste YouTube link:")
user_question = st.text_input("Your question:")

if youtube_link and user_question:
    try:
        # Download YouTube video
        yt = YouTube(youtube_link)
        video_stream = yt.streams.filter(only_audio=True).first()
        video_path = video_stream.download()

        # Transcribe video using OpenAI Whisper
        st.write("Transcribing video...")
        transcription = openai.Transcription.create(
            model="whisper",
            audio=video_path,
            return_text=True
        )

        # Check if transcription is empty or None
        if not transcription or not transcription.text:
            raise ValueError("Transcription is empty or None")

        # Chat with GPT-3
        st.write("Chatting with the video...")
        prompt = f"Q: {user_question}\nA:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )

        # Check if response is empty or None
        if not response or not response.choices:
            raise ValueError("GPT-3 response is empty or None")

        # Display results
        st.header("Transcription:")
        st.audio(video_path, format="audio/ogg")
        st.write(transcription.text.strip())

        st.header("Chat Response:")
        st.write(response.choices[0].text.strip())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
