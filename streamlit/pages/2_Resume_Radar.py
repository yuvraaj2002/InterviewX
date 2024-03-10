import streamlit as st
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from nltk.stem import WordNetLemmatizer
import io
import string
import nltk
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
def load_model():
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key="AIzaSyBmdKcpl5PTuVBVsHsOdAlotEOOAInFxoU"
    )
    return llm


def get_questions(text):

    llm = load_model()
    template = """
            After reviewing the comprehensive details {resume_content} outlined in the candidate's resume, 
            please provide five tailored questions that the candidate can anticipate, focusing specifically 
            on their projects and experience. 
            """

    prompt = PromptTemplate(
        input_variables=["resume_content"], template=template
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    response = chain.invoke(
        {"resume_content": text}
    )
    return response


@st.cache_resource
def load_w2v():
    W2V_model = KeyedVectors.load_word2vec_format(
        "artifacts/GoogleNews-vectors-negative300.bin.gz", binary=True, limit=500000
    )
    return W2V_model


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


# Define a function to get the vector representation of a document using Word2Vec
def vectorize_text(doc, w2v_model):

    # Remove out-of-vocabulary words and get Word2Vec vectors for the words in the document
    words = [word for word in doc.split() if word in w2v_model]
    if not words:
        # If none of the words are in the Word2Vec model, return zeros
        return np.zeros(300)

    # Return the mean of Word2Vec vectors for words in the document
    return np.mean(w2v_model[words], axis=0)


def extract_clean_pdf(pdf_data):

    file_object = io.BytesIO(pdf_data)  # Create a BytesIO object
    reader = PdfReader(file_object)

    # Extract text from all pages
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return clean_data(text)


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (np.array): First vector.
        vec2 (np.array): Second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    return 1 - cosine(vec1, vec2)


def resume_radar_page():

    col1, col2 = st.columns(spec=(2, 1.3), gap="large")
    with col1:
        st.markdown(
            "<h1 style='text-align: left; font-size: 50px; '>Resume Radar👨‍💼</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 18px; text-align: left;'>This module is dedicated solely to the refinement of your resume, the analysis of its alignment with a specific job description, and interview preparation. Through this process, we aim to ensure optimal coherence and relevance between your resume and the targeted job role, as well as equip you with the top 5 interview questions you can expect.</p>",
            unsafe_allow_html=True,
        )
        job_description = st.text_input("Enter the job description")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    with col2:

        if uploaded_file is not None:
            pdf_data = uploaded_file.read()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<embed src="data:application/pdf;base64,{b64_pdf}" width="690" height="740" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.write("")
            analyze_bt = st.button("Analyze my resume🔎", use_container_width=True)
            if analyze_bt:
                with col1:
                    w2v_model = load_w2v()
                    clean_text_resume = extract_clean_pdf(pdf_data)
                    clean_text_jd = clean_data(job_description)

                    resume_vector = vectorize_text(clean_text_resume, w2v_model)
                    jd_vector = vectorize_text(clean_text_jd, w2v_model)
                    similarity_score = np.round(
                        cosine_similarity(resume_vector, jd_vector), 2
                    )
                    st.write(
                        "<p style='font-size: 22px;text-align: center;background-color:#E0FFFF;'>Your resume and job description have <strong>"
                        + str(similarity_score * 100)
                        + "% similarity</strong></p>",
                        unsafe_allow_html=True,
                    )
                    questions = get_questions(clean_text_resume)
                    st.write(questions)




resume_radar_page()
