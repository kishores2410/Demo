import streamlit as st
from agent import *

# Set the title of the app
st.title("Video Transcription and Chatbot App")

# Create a text input widget for entering the OpenAI API key
api_key = st.text_input("Enter OpenAI API key")
set_key_button = st.button("Set API Key")

# When the "Set API Key" button is clicked, set the OpenAI API key
if set_key_button:
    set_apikey(api_key)
    st.success("OpenAI API key is set")

# Initialize variables for video and chat history
video_url = ""
video_path = ""
chat_history = []

# Load and transcribe video
if st.button("Load Video"):
    video_url = st.text_input("Enter YouTube Video URL")
    video_path = load_video(video_url)
    st.video(video_path)

if st.button("Transcribe Video"):
    if not video_path:
        st.warning("Please load a video first.")
    else:
        result = process_video(video_path)
        st.write("Transcription Result:")
        st.write(result)

# Chatbot interaction
if st.button("Ask Question"):
    user_query = st.text_area("Ask a question")
    if not video_url:
        st.warning("Please load a video first.")
    elif not user_query:
        st.warning("Please enter a question.")
    else:
        responses = QuestionAnswer(chat_history, query=user_query, url=video_url)
        chat_history.append((user_query, responses[-1]))
        st.write("Chatbot Response:")
        st.write(responses[-1])

# Reset the app
if st.button("Reset App"):
    reset_vars()
    st.success("App has been reset")
