import os
import tempfile
import streamlit as st
from streamlit_chat import message
from agent import Agent

st.set_page_config(
    page_title="பழைய இசை சந்தோஷம்",  # Change the page title to Tamil
    page_icon="🎵",  # Change the page icon to a music note emoji
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for design
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
        font-family: "TamilFont", sans-serif;  # Use a Tamil font for the text
    }
    .stTextInput {
        font-size: 16px;
    }
    .stButton {
        background-color: #FFD700;  # Change the button color to gold
        color: black;  # Change the button text color to black
    }
    .stButton:hover {
        background-color: #FFA500;  # Change the button hover color to orange
    }
    </style>
    """,
    unsafe_allow_html=True
)

def display_messages():
    st.subheader("அரட்டை")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"சிந்திக்கின்றேன்"):
            agent_text = st.session_state["agent"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["agent"].forget()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Pass the PDF file name to the ingest method
        pdf_name = file.name
        with st.session_state["ingestion_spinner"], st.spinner(f"உட்கொண்டு வாங்குகின்றேன் {pdf_name}"):
            st.session_state["agent"].ingest(file_path, pdf_name)
        os.remove(file_path)

def is_openai_api_key_set() -> bool:
    return len(st.session_state["OPENAI_API_KEY"]) > 0

def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        if is_openai_api_key_set():
            st.session_state["agent"] = Agent(st.session_state["OPENAI_API_KEY"])
        else:
            st.session_state["agent"] = None

    st.header("பழைய இசை சந்தோஷம்")  # Change the header to Tamil

    if st.text_input("OpenAI API கீ", value=st.session_state["OPENAI_API_KEY"], key="input_OPENAI_API_KEY", type="password"):
        if (
            len(st.session_state["input_OPENAI_API_KEY"]) > 0
            and st.session_state["input_OPENAI_API_KEY"] != st.session_state["OPENAI_API_KEY"]
        ):
            st.session_state["OPENAI_API_KEY"] = st.session_state["input_OPENAI_API_KEY"]
            if st.session_state["agent"] is not None:
                st.warning("தயவுசெய்து, கோப்புகளை மீண்டும் பதிவேற்றவும்.")
            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            st.session_state["agent"] = Agent(st.session_state["OPENAI_API_KEY"])

    st.subheader("மாதிரி இசை கேள்க்க")  # Change the subheader to Tamil
    st.write("- குரலின் அம்சங்கள் என்ன?")
    st.write("- எப்படி இசைக்கு மேலும் அறிந்தது?")
    st.write("- திரைப்படத்தில் அம்சம் பார்த்தால் என்ன அந்த குரல் அம்சம் என்ன?")

    st.subheader("இசை ஆவணம் பதிவேற்ற")  # Change the subheader to Tamil
    st.file_uploader(
        "இசை ஆவணம் பதிவேற்ற (PDF)",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_openai_api_key_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("இசை கேட்க கேள்க்கவும்", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)

    st.divider()

if __name__ == "__main__":
    main()
