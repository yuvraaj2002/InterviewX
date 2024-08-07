import streamlit as st
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import tensorflow_hub as hub
import io
import os
import string
import nltk
import time
import numpy as np
from nltk.corpus import wordnet
import re
import gensim
from gensim.models import Word2Vec, KeyedVectors
from scipy.spatial.distance import cosine
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


nltk.download("stopwords")
import string

nltk.download("punkt")
punctuation = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words("english"))


@st.cache_resource
def load_models():
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=os.getenv("GEMINI_API_KEY")
    )
    embedding_model = KeyedVectors.load_word2vec_format(
        "artifacts/GoogleNews-vectors-negative300.bin.gz", binary=True, limit=500000
    )
    return llm, embedding_model


def get_questions(text,llm):

    template = """
            After reviewing the comprehensive details {resume_content} outlined in the candidate's resume, 
            please provide seven tailored questions that the candidate can anticipate, focusing specifically 
            on their projects and experience.Return me python list of questions which I could iterate.
            """

    prompt = PromptTemplate(input_variables=["resume_content"], template=template)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    response = chain.invoke({"resume_content": text})
    return response



def clean_data(text):
    """
    This function preprocesses text by performing the following steps:

    1. **Lowercasing:** Converts all characters to lowercase.
    2. **Tokenization:** Splits the text into individual words using NLTK word tokenization.
    3. **Stopword removal:** Removes common stop words (e.g., "the", "a", "is") from the word list.
    4. **Punctuation removal:** Removes punctuation marks from the word list.
    5. **Remove phone numbers and email addresses:** Removes phone numbers and email addresses from the text.
    6. **Lemmatization:** Lemmatizes words to their base form.
    7. **Rejoining:** Joins the remaining words back into a string with spaces in between.
    


    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text with stop words, punctuation, phone numbers, email addresses, and lemmatization applied.
    """
    # Lowercase the text
    text_lower = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text_lower)

    # Remove punctuation marks
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove phone numbers
    cleaned_tokens = []
    for token in tokens:
        if not re.match(r"\+?\d[\d -]{8,12}\d", token):
            cleaned_tokens.append(token)

    # Remove "gmail.com" after punctuation removal
    final_tokens = []
    for token in cleaned_tokens:
        if not token.endswith("gmail.com"):
            final_tokens.append(token)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in final_tokens]

    # Rejoin the tokens into a string
    cleaned_text = " ".join(lemma_tokens)
    return cleaned_text



def extract_clean_pdf(pdf_data):

    file_object = io.BytesIO(pdf_data)  # Create a BytesIO object
    reader = PdfReader(file_object)

    # Extract text from all pages
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return clean_data(text)


# Define a function to get the vector representation of a document using Word2Vec
def vectorize_text(doc, embedding_model):

    # Remove out-of-vocabulary words and get Word2Vec vectors for the words in the document
    words = [word for word in doc.split() if word in embedding_model]
    if not words:
        # If none of the words are in the Word2Vec model, return zeros
        return np.zeros(300)

    # Return the mean of Word2Vec vectors for words in the document
    return np.mean(embedding_model[words], axis=0)



def download_questions(questions_text):
    """
    Creates a text file with the given summary text and offers download.
    """
    # Generate unique filename
    filename = f"Interview_Questions_{int(time.time())}.txt"

    # Create the file and write the content
    with open(filename, "w") as file:
        file.write(questions_text)

    # Set content_type and headers
    content_type = "text/plain"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Content-type": content_type,
    }

    # Use st.download_button to offer download
    st.download_button(
        "Download questions💾",
        data=questions_text,
        file_name=filename,
        mime=content_type,
        use_container_width=True,
    )


# State management
if 'clean_text_resume' not in st.session_state:
    st.session_state.clean_text_resume = None


def resume_radar_page():
    llm, embedding_model = load_models()
    col1, col2 = st.columns(spec=(2, 1.3), gap="large")
    uploaded_file = None
    with col1:
        st.markdown(
            "<h1 style='text-align: left; font-size: 50px; '>Resume Radar👨‍💼</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;'>We are excited to introduce our innovative module designed to streamline and optimize the job application process. This module allows users to upload their resumes and input a job description for the position they are applying for. Leveraging the power of advanced AI algorithms, our system meticulously analyzes the content of the uploaded resume and compares it against the provided job description. The AI-driven analysis generates a similarity score that offers valuable insights into how well the resume aligns with the job requirements. This score helps users identify strengths and areas for improvement, enhancing their chances of securing the desired position.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;'>In addition to the similarity score, and with the user's consent, our module goes a step further by providing a tailored set of interview preparation tools. Specifically, it generates a list of the top 10 interview questions that candidates can expect, based on the unique content of their resume. These questions are designed to prepare candidates effectively, ensuring they are well-equipped to showcase their skills and experience during the interview.</p>",
            unsafe_allow_html=True,
        )
        job_description = st.text_input("Enter the job description")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    with col2:
        if uploaded_file is not None:
            pdf_data = uploaded_file.read()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            st.write("")
            pdf_display = f'<embed src="data:application/pdf;base64,{b64_pdf}" width="690" height="740" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

            analyze_bt = st.button("Analyze my resume🔎", use_container_width=True)
            if analyze_bt:
                if not job_description:
                    with col1:
                        st.error("Please provide the job description first.")
                else:
                    with col1:
                        clean_text_resume = extract_clean_pdf(pdf_data)
                        clean_text_jd = clean_data(job_description)

                        # Generate embeddings for resume and job description
                        resume_vector = vectorize_text(clean_text_resume, embedding_model)
                        jd_vector = vectorize_text(clean_text_jd, embedding_model)
                        similarity_score = np.round(1 - cosine(resume_vector, jd_vector),2)
                    
                        st.write(
                            "<p style='font-size: 22px;text-align: center;background-color:#C3E8FF;'>Your resume and job description have <strong>"
                            + str(similarity_score * 100)
                            + "% similarity</strong></p>",
                            unsafe_allow_html=True,
                        )
                        # Save clean_text_resume in session state
                        st.session_state.clean_text_resume = clean_text_resume

    concent_button = st.button("Generate tailored Interview questions as per your resume", use_container_width=True)
    if concent_button:
        if st.session_state.clean_text_resume:
            questions = get_questions(st.session_state.clean_text_resume,llm)
            download_questions(questions)
            st.write("***")
        else:
            st.error("Please analyze your resume first.")


resume_radar_page()


