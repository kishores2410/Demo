import os
import tempfile
import whisper
from typing import Union 
import datetime as dt
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pytube import YouTube
from typing import Any, Generator, List

# Set the Streamlit layout to wide
st.set_page_config(layout="wide")

# Streamlit widgets
st.title("Video Transcription and Chatbot")

# Sidebar widgets
st.sidebar.header("API Key Management")
api_key = st.sidebar.text_input("Enter OpenAI API key")
api_key_set = st.sidebar.checkbox("Set API Key")
api_key_remove = st.sidebar.checkbox("Remove API Key")

if api_key_set:
    os.environ['OPENAI_API_KEY'] = api_key
    st.sidebar.text("API Key is set")
elif api_key_remove:
    os.environ['OPENAI_API_KEY'] = ''
    st.sidebar.text("API Key removed")

# Streamlit widgets
st.sidebar.subheader("Video Transcription")
video_url = st.sidebar.text_input("Enter YouTube URL")
transcribe_button = st.sidebar.button("Initiate Transcription")

st.sidebar.subheader("Chatbot")
query = st.sidebar.text_area("Enter your query here")
ask_button = st.sidebar.button("Ask Chatbot")

# Global variables
chat_history = []
result = None
chain = None
run_once_flag = False
call_to_load_video = 0

enable_box = st.empty()
disable_box = st.empty()
remove_box = st.empty()
pause = st.empty()
resume = st.empty()
update_video = st.empty()
update_yt = st.empty()

def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    disable_box.text('OpenAI API key is Set')
    enable_box.empty()
    return

def enable_api_box():
    enable_box.textbox('Upload your OpenAI API key', value=None)
    disable_box.empty()
    remove_box.empty()

def remove_key_box():
    os.environ['OPENAI_API_KEY'] = ''
    remove_box.text('Your API key successfully removed')
    enable_box.empty()
    disable_box.empty()
    return

def reset_vars():
    global chat_history, result, chain, run_once_flag, call_to_load_video

    os.environ['OPENAI_API_KEY'] = ''
    chat_history = []
    result, chain = None, None
    run_once_flag, call_to_load_video = False, 0

def load_video(url:str) -> str:
    global result

    yt = YouTube(url)
    target_dir = os.path.join(tempfile.gettempdir(), 'Youtube')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if os.path.exists(os.path.join(target_dir, f"{yt.title}.mp4")):
        return os.path.join(target_dir, f"{yt.title}.mp4")
    try:
        stream = yt.streams.filter(only_audio=True).first()
        print('----DOWNLOADING AUDIO FILE----')
        stream.download(output_path=target_dir)
    except:
        raise st.error('Issue in Downloading video')

    return os.path.join(target_dir, f"{yt.title}.mp4")

def process_video(video=None, url=None) -> dict[str, Union[str, list]]:

    if url:
        file_dir = load_video(url)
    else:
        file_dir = video

    st.write('Transcribing Video with whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)

    return result

def process_text(video=None, url=None) -> tuple[list, list[dt.datetime]]:
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

def get_title(url, video):
    if url != None:
        yt = YouTube(url)
        title = yt.title
    else:
        title = os.path.basename(video)
        title = title[:-4]
    return title

def check_path(url=None, video=None):
    if url:
        yt = YouTube(url)
        if os.path.exists(os.path.join(tempfile.gettempdir(), 'Youtube', f"{yt.title}.mp4")):
            return True
    else:
        if os.path.exists(video):
            return True
    return False

def make_chain(url=None, video=None) -> Union[ConversationalRetrievalChain, Any, None]:
    global chain, run_once_flag

    if not url and not video:
        st.error('Please provide a Youtube link or Upload a video')
        return None
    if not run_once_flag:
        run_once_flag=True
        title = get_title(url, video).replace(' ','-')

        # if not check_path(url, video):
        grouped_texts, time_list = process_text(url=url) if url else process_text(video=video)
        time_list = [{'source':str(t.time())} for t in time_list]

        vector_stores = Chroma.from_texts(texts=grouped_texts,collection_name= 'test',embedding=OpenAIEmbeddings(), metadatas=time_list)
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.0), 
                                                retriever=vector_stores.as_retriever(search_kwargs={"k": 5}),
                                                return_source_documents=True )

        return chain
    else:
        return chain

def QuestionAnswer(history, query=None, url=None, video=None) -> Generator[Union[Any, None], Any, None]:
    global chat_history, chain

    if video and url:
        st.error('Upload a video or a Youtube link, not both')
        return

    if not url and not video:
        st.error('Provide a Youtube link or Upload a video')
        return

    result = chain({"question": query, 'chat_history':chat_history},return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    for char in result['answer']:
        history[-1][-1] += char
        yield history,''

def add_text(history, text):
    if not text:
         st.error('Enter text')
         return
    history = history + [(text,'')]
    return history

def embed_yt(yt_link: str):
    global run_once_flag, call_to_load_video

    # Check if the YouTube link is valid.
    if not yt_link:
        st.error('Paste a Youtube link')
        return

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

def embed_video(video=str | None):
    global run_once_flag

    # Check if the video is valid.
    if not video:
        st.error('Upload a Video')
        return

    # Set the global variable `run_once_flag` to False.
    # This is used to prevent the function from being called more than once.
    run_once_flag = False

    # Create a chain using the video.
    make_chain(video=video)

    # Return the video and an empty list.
    return video, []

if transcribe_button:
    st.write("Transcribing Video...")
    video_file = load_video(video_url)
    result = process_video(video_file)

if ask_button:
    st.write("Asking Chatbot...")
    chat_responses = list(QuestionAnswer(chat_history, query))
    chat_history = chat_responses[0][0]

# Display the video or chat history
if result:
    st.subheader("Video Transcription")
    st.video(result)  # Display the video here

if chat_history:
    st.subheader("Chat History")
    for question, answer in chat_history:
        st.write(f"User: {question}")
        st.write(f"Chatbot: {answer}")
