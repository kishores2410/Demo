import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

class Agent:
    def __init__(self, openai_api_key: str = None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

        self.chat_history = None
        self.chain = None
        self.db = None
        self.current_pdf = None  # To keep track of the currently ingested PDF
    
    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            response = self.chain({"question": question, "chat_history": self.chat_history})
            response = response["answer"].strip()
    
            # Check if the response contains "I don't know" and replace it with the desired message
            if "I don't know." in response:
                response = "Sorry, I am yet to be trained on this topic. Please try some other question related to the uploaded file."
            else:
                # Include the reference to the PDF file if available
                if self.current_pdf:
                    response += f" (Ref: {self.current_pdf})"
                
            self.chat_history.append((question, response))
        return response




    def ingest(self, file_path: os.PathLike, pdf_name: str) -> None:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        
        self.current_pdf = pdf_name  # Store the PDF file name
        
        if self.db is None:
            self.db = FAISS.from_documents(splitted_documents, self.embeddings)
            self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever())
            self.chat_history = []
        else:
            self.db.add_documents(splitted_documents)

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history = None
        self.current_pdf = None
