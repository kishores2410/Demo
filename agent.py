import os
import tempfile
import whisper
import datetime as dt
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pytube import YouTube
from typing import Any, List, Union


# Global variables
chat_history = []
result = None
chain = None
run_once_flag = False
call_to_load_video = 0

# Function to set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key

# Function to load a video from a URL
def load_video(url: str) -> str:
    yt = YouTube(url)
    target_dir = os.path.join('/tmp', 'Youtube')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if os.path.exists(target_dir+'/'+yt.title+'.mp4'):
        return target_dir+'/'+yt.title+'.mp4'
    try:
        yt.streams.filter(only_audio=True)
        stream = yt.streams.get_audio_only()
        print('----DOWNLOADING AUDIO FILE----')
        stream.download(output_path=target_dir)
    except:
        raise Exception('Issue in Downloading video')

    return target_dir+'/'+yt.title+'.mp4'

def process_video(video=None, url=None) -> Union[dict[str, str], dict[str, list]]:
    if url:
        file_dir = load_video(url)
    else:
        file_dir = video

    print('Transcribing Video with whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)
    return result

# Function to process video text and extract timestamps
def process_text(video=None, url=None) -> tuple[list, list[dt.datetime]]:
    global call_to_load_video

    if call_to_load_video == 0:
        print('yes')
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

    if current_group:
        grouped_texts.append(current_group)

    return grouped_texts, time_list

# Function to get the title of a video from URL or file path
def get_title(url, video):
    if url != None:
        yt = YouTube(url)
        title = yt.title
    else:
        title = os.path.basename(video)
        title = title[:-4]
    return title

# Function to check if a video file exists
def check_path(url=None, video=None):
    if url:
        yt = YouTube(url)
        if os.path.exists('/tmp/Youtube' + yt.title + '.mp4'):
            return True
    else:
        if os.path.exists(video):
            return True
    return False

# Function to create a Conversational Retrieval Chain
def make_chain(url=None, video=None) -> (ConversationalRetrievalChain | Any | None):
    global chain, run_once_flag

    if not url and not video:
        raise Exception('Please provide a Youtube link or Upload a video')
    if not run_once_flag:
        run_once_flag = True
        title = get_title(url, video).replace(' ', '-')

        grouped_texts, time_list = process_text(url=url) if url else process_text(video=video)
        time_list = [{'source': str(t.time())} for t in time_list]

        vector_stores = Chroma.from_texts(texts=grouped_texts, collection_name='test', embedding=OpenAIEmbeddings(),
                                          metadatas=time_list)
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.0),
                                                      retriever=vector_stores.as_retriever(search_kwargs={"k": 5}),
                                                      return_source_documents=True)

        return chain
    else:
        return chain

# Function to handle chatbot questions and answers
def QuestionAnswer(history, query=None, url=None, video=None) -> List[str]:
    global chat_history, chain

    if video and url:
        raise Exception('Upload a video or a Youtube link, not both')
    elif not url and not video:
        raise Exception('Provide a Youtube link or Upload a video')

    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''

# Function to add text to chat history
def add_text(history, text):
    if not text:
        raise Exception('Enter text')
    history = history + [(text, '')]
    return history

# Function to reset app variables
def reset_vars():
    global chat_history, result, chain, run_once_flag, call_to_load_video

    os.environ['OPENAI_API_KEY'] = ''
    chat_history = None
    result, chain = None, None
    run_once_flag, call_to_load_video = False, 0
