import pandas as pd
import numpy as np
import streamlit as st
import whisper
import pytube
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os
import pinecone
from dotenv import load_dotenv

# Initialize Streamlit app
st.markdown('<h1>Youtube GPT 🤖<small> by <a href="https://codegpt.co">Code GPT</a></small></h1>', unsafe_allow_html=True)

# Input field for YouTube link
youtube_link = st.text_input(label="Paste YouTube link here:", value="https://youtu.be/rQeXGvFAJDQ")

# Check if a YouTube link is provided
if youtube_link:
    # Split the YouTube URL to extract video ID
    video_id = pytube.extract.video_id(youtube_link)
    
    # Whisper
    model = whisper.load_model('base')
    output = model.transcribe(youtube_link)
    
    # Transcription
    transcription = {
        "title": "Video Title",  # You can extract the title from the YouTube video object
        "transcription": output['text']
    }
    
    # Display transcription
    st.header("Transcription:")
    st.write(transcription["transcription"])
    
    # Pinacone
    load_dotenv()
    # Initialize connection to Pinecone
    pinecone.init(
        api_key=os.getenv("PINACONE_API_KEY"),
        environment=os.getenv("PINACONE_ENVIRONMENT")
    )
    
    # Embeddings
    data = []
    for segment in output['segments']:
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input=segment["text"].strip(),
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        meta = {
            "text": segment["text"].strip(),
            "start": segment['start'],
            "end": segment['end'],
            "embedding": embeddings
        }
        data.append(meta)
    
    # Create or update Pinecone index
    index_name = str(video_id)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(data[0]['embedding']))
    
    # Connect to index
    index = pinecone.Index(index_name)
    
    # Upsert embeddings into Pinecone
    upsert_response = index.upsert(
        vectors=data,
        namespace=video_id
    )
    
    # Display the embeddings
    st.header("Embedding:")
    df = pd.DataFrame(data)
    st.write(df)
