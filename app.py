import os
import tempfile
import datetime as dt
import whisper
from pytube import YouTube
import streamlit as st
from typing import Optional


# Initialize variables
chat_history = []
result = None
chain = None
run_once_flag = False
call_to_load_video = 0

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("Video Transcription and Chatbot")

# API Key Input
api_key = st.text_input("Enter OpenAI API key", key="api_key")

# Change Key Button
if st.button("Change Key"):
    os.environ['OPENAI_API_KEY'] = api_key
    st.success("OpenAI API key is set.")

# Remove Key Button
if st.button("Remove Key"):
    os.environ['OPENAI_API_KEY'] = ''
    st.success("Your API key has been successfully removed.")

# Video Input
st.sidebar.subheader("Video Input")
video_option = st.sidebar.radio("Choose Video Source", ("Upload Video", "YouTube URL"))

if video_option == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
else:
    yt_link = st.sidebar.text_input("Paste a YouTube link")
    if st.sidebar.button("Initiate Transcription"):
        # Call the process_video function with the YouTube link
        result = process_video(url=yt_link)

# Embed Video or YouTube
st.sidebar.subheader("Embed Video")
if st.sidebar.button("Embed Video"):
    st.write("Embed the video here")

# Reset App Button
if st.sidebar.button("Reset App"):
    # Call the reset_vars function
    chat_history, result, chain, run_once_flag, call_to_load_video = reset_vars()

# Chatbot Interface
st.subheader("Chatbot Interface")
query = st.text_input("Enter your query here")

if st.button("Ask Question"):
    # Call the QuestionAnswer function
    for response, _ in QuestionAnswer(chat_history, query, yt_link, uploaded_video):
        # Display responses
        st.write(response[-1][-1])

# Display chat history
st.write("Chat History:")
for item in chat_history:
    st.write(f"{item[0]}: {item[1]}")

# Function to load video and transcribe it
def process_video(video=None, url=None):
    global result

    if url:
        file_dir = load_video(url)
    else:
        file_dir = video

    st.write('Transcribing Video with Whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)

# Function to process video and return transcribed text
def process_text(video=None, url=None):
    global call_to_load_video

    if call_to_load_video == 0:
        st.write('yes')
        result = process_video(url=url) if url else process_video(video=video)
        call_to_load_video += 1

    texts, start_time_list = [], []

    for res in result['segments']:
        start = res['start']
        text = res['text']

        start_time = dt.datetime.fromtimestamp(start)
        start_time_formatted = start_time.strftime("%H:%M:%S")

        texts.append(''.join(text))
        start_time_list.append(start_time_formatted)

    texts_with_timestamps = dict(zip(texts, start_time_list))
    formatted_texts = {
        text: dt.datetime.strptime(str(timestamp), '%H:%M:%S')
        for text, timestamp in texts_with_timestamps.items()
    }

    grouped_texts = []
    current_group = ''
    time_list = [list(formatted_texts.values())[0]]
    previous_time = None
    time_difference = dt.timedelta(seconds=30)

    for text, timestamp in formatted_texts.items():

        if previous_time is None or timestamp - previous_time <= time_difference:
            current_group += text
        else:
            grouped_texts.append(current_group)
            time_list.append(timestamp)
            current_group = text
        previous_time = time_list[-1]

    # Append the last group of texts
    if current_group:
        grouped_texts.append(current_group)

    return grouped_texts, time_list

# Function to load video from YouTube and return the file path
def load_video(url:str) -> str:
    global result

    yt = YouTube(url)
    target_dir = os.path.join('/tmp', 'Youtube')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if os.path.exists(target_dir+'/'+yt.title+'.mp4'):
        return target_dir+'/'+yt.title+'.mp4'
    try:

        yt.streams.filter(only_audio=True)
        stream = yt.streams.get_audio_only()
        st.write('----DOWNLOADING AUDIO FILE----')
        stream.download(output_path=target_dir)
    except:
        st.error('Issue in Downloading video')

    return target_dir+'/'+yt.title+'.mp4'

# Function to set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box

# Function to enable the API key input box
def enable_api_box():
    return enable_box

# Function to remove the OpenAI API key
def remove_key_box():
    os.environ['OPENAI_API_KEY'] = ''
    return remove_box

# Function to reset app variables
def reset_vars():
    global chat_history, result, chain, run_once_flag, call_to_load_video

    os.environ['OPENAI_API_KEY'] = ''
    chat_history = None
    result, chain = None, None
    run_once_flag, call_to_load_video = False, 0

    return [], '', None, None, None

# Function to create a ConversationalRetrievalChain
def make_chain(url=None, video=None):
    global chain, run_once_flag

    if not url and not video:
        st.error('Please provide a Youtube link or Upload a video')
    if not run_once_flag:
        run_once_flag = True
        title = get_title(url, video).replace(' ','-')

        grouped_texts, time_list = process_text(url=url) if url else process_text(video=video)
        time_list = [{'source': str(t.time())} for t in time_list]

        vector_stores = Chroma.from_texts(texts=grouped_texts,collection_name= 'test',embedding=OpenAIEmbeddings(), metadatas=time_list)
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.0), 
                                                    retriever=vector_stores.as_retriever(search_kwargs={"k": 5}),
                                                    return_source_documents=True )

        return chain
    else:
        return chain

# Function to get the title of the video (either from URL or uploaded file)
def get_title(url, video):
    if url != None:
        yt = YouTube(url)
        title = yt.title
    else:
        title = os.path.basename(video)
        title = title[:-4]
    return title

# Function to check if the video file exists
def check_path(url=None, video=None):
    if url:
        yt = YouTube(url)   
        if os.path.exists('/tmp/Youtube'+yt.title+'.mp4'):
            return True
    else:
        if os.path.exists(video):
            return True
    return False

# Function to embed a YouTube video
def embed_yt(yt_link: str):
    # This function embeds a YouTube video into the page.

    # Check if the YouTube link is valid.
    if not yt_link:
        st.error('Paste a Youtube link')

    # Set the global variable `run_once_flag` to False.
    # This is used to prevent the function from being called more than once.
    run_once_flag = False

    # Set the global variable `call_to_load_video` to 0.
    # This is used to keep track of how many times the function has been called.
    call_to_load_video = 0

    # Create a chain using the YouTube link.
    make_chain(url=yt_link)

    # Get the URL of the YouTube video.
    url = yt_link.replace('watch?v=', '/embed/')

    # Create the HTML code for the embedded YouTube video.
    embed_html = f"""<iframe width="750" height="315" src="{url}"
                     title="YouTube video player" frameborder="0"
                     allow="accelerometer; autoplay; clipboard-write;
                     encrypted-media; gyroscope; picture-in-picture"
                     allowfullscreen></iframe>"""

    # Return the HTML code and an empty list.
    return embed_html, []

# # Function to embed a video
# def embed_video(video=str | None):
#     # This function embeds a video into the page.

#     # Check if the video is valid.
#     if not video:
#         st.error('Upload a Video')

#     # Set the global variable `run_once_flag` to False.
#     # This is used to prevent the function from being called more than once.
#     run_once_flag = False

#     # Create a chain using the video.
#     make_chain(video=video)

#     # Return the video and an empty list.
#     return video, []

def embed_video(video: Optional[str] = None):
    # This function embeds a video into the page.

    # Check if the video is valid.
    if not video:
        st.error('Upload a Video')

    # Set the global variable `run_once_flag` to False.
    # This is used to prevent the function from being called more than once.
    run_once_flag = False

    # Create a chain using the video.
    make_chain(video=video)

    # Return the video and an empty list.
    return video, []


# Streamlit app entry point
if __name__ == "__main__":
    streamlit run app.py
    st.run()
    
