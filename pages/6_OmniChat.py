import streamlit as st
import os
import io 
import base64
import time
import re
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader

from together import Together
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
import numpy as np

from dotenv import load_dotenv
load_dotenv()



st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.8rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


# Function to extract the video ID from the YouTube URL
def get_youtube_id(url):
    
    regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    match = re.match(regex, url)
    return match.group(6) if match else None




class TogetherEmbeddingWrapper(Embeddings):

    def __init__(self, api_key, model_api_string):
        self.client = Together(api_key=api_key)
        self.model_api_string = model_api_string

    def embed_documents(self, texts):
        outputs = self.client.embeddings.create(input=texts, model=self.model_api_string)
        return [x.embedding for x in outputs.data]  # Return embeddings

    def embed_query(self, text):
        outputs = self.client.embeddings.create(input=[text], model=self.model_api_string)
        return [x.embedding for x in outputs.data][0]  # Return single embedding


@st.cache_resource
def load_models():
    api_key = os.environ['TOGETHER_API_KEY']  
    model_api_string = "togethercomputer/m2-bert-80M-8k-retrieval"  
    embedding_model = TogetherEmbeddingWrapper(api_key, model_api_string)

    # Initialize Together client
    client = Together(api_key=api_key)
    return client, embedding_model  # Return client instead of llm





def download_trasncript(text):
    """
    Creates a text file with the given summary text and offers download.
    """
    # Generate unique filename
    filename = f"summary_{int(time.time())}.txt"

    # Create the file and write the content
    with open(filename, "w") as file:
        file.write(text)

    # Set content_type and headers
    content_type = "text/plain"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Content-type": content_type,
    }

    # Use st.download_button to offer download
    st.download_button(
        "Download Video transcript📝",
        data=text,
        file_name=filename,
        mime=content_type,
        use_container_width=True,
    )


def get_answer(vdb_contxt_text, query_text, client):
    messages = [
        {"role": "user", "content": f"Given the query '{query_text}', and after reviewing the information retrieved from the vector database: {vdb_contxt_text}, please provide a concise and informative answer."}
    ]
    
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        messages=messages,
    )
    return response.choices[0].message.content


def process_load(pdf_data,youtube_id,llm,embedding_model):

    result = YouTubeTranscriptApi.get_transcript(youtube_id)
    yt_captions = ""
    for item in iter(result):
        yt_captions = yt_captions + item['text'] + ""
    

    # Create a BytesIO object
    file_object = io.BytesIO(pdf_data)  
    reader = PdfReader(file_object)

    # Extract text from all pages
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()


    context_data = yt_captions + pdf_text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(context_data)]
    
    pinecone = PineconeVectorStore.from_documents(docs, embedding_model, index_name="omnichat",pinecone_api_key=os.environ['PINECONE_API_KEY'])
    return pinecone


def chat_with_utube():

    # Calling the function to load the Whisper model, LLM and embedding model
    llm,embedding_model = load_models()
    pinecone_obj = None

    col1, col2 = st.columns(spec=(2.5,1), gap="large")
    with col1:
        st.markdown(
            "<h1 style='text-align: left; font-size: 48px;'>OmniChat 🤖</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>Welcome to our advanced Media Query Module, a versatile tool for interacting with both YouTube videos and PDF documents. Upload any YouTube video URL or PDF file, and our AI will generate searchable transcripts and text extractions. These are stored in our vector database, linked to a Large Language Model (LLM), enabling detailed content queries. Retrieve insights effortlessly with our cutting-edge AI technology.</p>",
            unsafe_allow_html=True,
        )

        # Getting the Youtube URL as input from the user
        with st.container(border=True):
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            youtube_link = st.text_input("Enter the Youtube video URL")

    
    with col2:
        if youtube_link and pdf_file:
            video_id = get_youtube_id(youtube_link)
            if video_id:
                st.write("")
                with st.container(border=True):
                    st.markdown(f"""
                        <iframe width="430" height="280" src="https://www.youtube.com/embed/{video_id}" 
                        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen></iframe>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    pdf_data = pdf_file.read()
                    b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
                    pdf_display = f'<embed src="data:application/pdf;base64,{b64_pdf}" width="430" height="500" type="application/pdf">'
                    st.markdown(pdf_display, unsafe_allow_html=True)

                    # Calling the function to process both PDF and Youtube and store them in Vector DB
                    pinecone_obj = process_load(pdf_data,video_id,llm,embedding_model)
            else:
                st.error("Invalid YouTube URL")
        else:
            st.write("")
            st.error("Please provide both a YouTube link and a PDF file")

    
 
    if pinecone_obj:
        response = None
        with col1:
            message = st.chat_message("assistant")
            message.write("Video is Processed Succesfully, you can start with your query")
            query_text = st.text_input("Enter your query : ")
            if query_text:
                result = pinecone_obj.similarity_search(query_text)[:1]
                vdb_context_text = result[0].page_content

                # Calling the function to get the answer from the LLM
                response = get_answer(vdb_context_text,query_text,llm)

            if response is not None:
                with st.container(border=True):
                    st.markdown(
                            f"<p style='font-size: 20px;'>{response}</p>",
                        unsafe_allow_html=True
                    )

            

chat_with_utube()