import os
import tempfile
import datetime as dt
import whisper
from pytube import YouTube
import streamlit as st

# Initialize variables
chat_history = []
result = None
chain = None
run_once_flag = False
call_to_load_video = 0

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("Video Transcription and Chatbot")

# Function to process video and return transcribed text
def process_video(video=None, url=None):
    global result

    if url:
        file_dir = load_video(url)
    else:
        file_dir = video

    st.write('Transcribing Video with Whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)

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

# Function to get the title of the video (either from URL or uploaded file)
def get_title(url, video):
    if url != None:
        yt = YouTube(url)
        title = yt.title
    else:
        title = os.path.basename(video)
        title = title[:-4]
    return title

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

# Function to reset app variables
def reset_vars():
    global chat_history, result, chain, run_once_flag, call_to_load_video

    os.environ['OPENAI_API_KEY'] = ''
    chat_history = None
    result, chain = None, None
    run_once_flag, call_to_load_video = False, 0

    return [], '', None, None, None

# Main Streamlit app
def main():
    global chain, chat_history

    with st.sidebar:
        #enable_box = st.textbox(label='Upload your OpenAI API key', value=None, key="enable_box")
        enable_box = st.text_input(label='Upload your OpenAI API key', value=None, key="enable_box")
        disable_box = st.text_area(label='OpenAI API key is Set', value='OpenAI API key is Set', key="disable_box")
       # disable_box = st.textbox(label='OpenAI API key is Set', value='OpenAI API key is Set', key="disable_box")
      #  remove_box = st.textbox(label='Your API key successfully removed', value='Your API key successfully removed', key="remove_box")
        remove_box = st.text("Your API key successfully removed")
        pause = st.button(label='Pause', key="pause")
        resume = st.button(label='Resume', key="resume")

    if st.sidebar.button("Change Key"):
        set_apikey(enable_box)
        st.success("OpenAI API key is set.")

    if st.sidebar.button("Remove Key"):
        remove_key_box()
        st.success("Your API key has been successfully removed.")

    if st.sidebar.button("Reset App"):
        chat_history, result, chain, run_once_flag, call_to_load_video = reset_vars()
        st.success("App has been reset.")

    with st.sidebar:
        st.write("Please reset the app after being done with the app to remove resources")

    st.sidebar.markdown("---")

    with st.sidebar:
        st.markdown("### Transcription and Chatbot")
        st.write("1. Upload a video file (e.g., MP4) or provide a YouTube link.")
        st.write("2. Click 'Initiate Transcription' to transcribe the video.")
        st.write("3. Ask questions or have a conversation with the chatbot.")
        st.write("4. Reset the app to remove resources.")

    st.sidebar.markdown("---")

    st.sidebar.text("Chat History")

    # Main content
    st.markdown("## Video Transcription and Chatbot")

    st.markdown("### Upload Video or Provide YouTube Link")

    col1, col2 = st.beta_columns(2)

    with col1:
        video = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

    with col2:
        yt_link = st.text_input("Paste a YouTube link here")

    st.write("---")

    if st.button("Initiate Transcription"):
        if video and yt_link:
            st.error("Upload a video or provide a YouTube link, but not both.")
        elif not video and not yt_link:
            st.error("Provide a YouTube link or upload a video.")
        else:
            if video:
                video_path = tempfile.mktemp(suffix=".mp4")
                with open(video_path, "wb") as f:
                    f.write(video.read())
                process_video(video=video_path)
            else:
                process_video(url=yt_link)

    st.write("---")

    if result:
        st.markdown("### Video Transcription")

        texts, time_list = process_text()
        for text, timestamp in zip(texts, time_list):
            st.write(f"{timestamp}: {text}")

        st.write("---")

        st.markdown("### Chat with the Chatbot")
        st.write("Ask questions or have a conversation with the chatbot.")

        chat_input = st.text_input("Enter your question or message")
        if st.button("Send"):
            if chat_input:
                response = chain({"question": chat_input, 'chat_history': chat_history}, return_only_outputs=True)
                chat_history.append((chat_input, response["answer"]))
                st.write(f"You: {chat_input}")
                st.write(f"Chatbot: {response['answer']}")
            else:
                st.error("Please enter a question or message.")

if __name__ == "__main__":
    main()
