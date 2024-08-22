import streamlit as st
from bs4 import BeautifulSoup
import requests
import time
import os
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.2rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)
os.environ["GROQ_API_KEY"] = os.getenv('groq_api_key')


@st.cache_resource
def laoding_models():
    
    # Loading the LLM model
    model = ChatGroq(model="llama3-8b-8192")

    # Embedding Model
    embedding_model = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv('HF_TOKEN'), model_name="BAAI/bge-base-en-v1.5")

    # Defining the semantic chunker
    text_splitter_semantic = SemanticChunker(
    embedding_model, breakpoint_threshold_type="percentile")

    return model,text_splitter_semantic


@st.cache_resource
def get_summary_chain(_model):

    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response containing exactly 3 bullet points that cover the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=_model,
                                        chain_type='map_reduce',
                                        combine_prompt=combine_prompt_template)
    
    return summary_chain



def download_summary(summary_text):
    """
    Creates a text file with the given summary text and offers download.
    """
    # Generate unique filename
    filename = f"summary_{int(time.time())}.txt"

    # Create the file and write the content
    with open(filename, "w") as file:
        file.write(summary_text)

    # Set content_type and headers
    content_type = "text/plain"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Content-type": content_type,
    }

    # Use st.download_button to offer download
    st.download_button(
        "Download Summary",
        data=summary_text,
        file_name=filename,
        mime=content_type,
        use_container_width=True,
    )


def extract_process(URL):
    try:
        # Extracting the html content from the given URL
        response = requests.get(URL)

        # Check if the request was successful
        if response.status_code == 200:

            # Instantiating BeautifulSoup class
            bsoup = BeautifulSoup(response.text, "html.parser")

            # Extracting the main article content
            article_content = bsoup.find_all(["h1", "p"])

            if article_content:

                # Extracting the text from paragraphs within the article
                text = [content.text for content in article_content]
                ARTICLE = " ".join(text)
                return ARTICLE

            else:
                return "Unable to extract article content. Please check the URL or website structure."
        else:
            return f"Failed to retrieve content. Status Code: {response.status_code}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"


def text_summarization_page():
    st.markdown(
        "<h1 style='text-align: left; font-size: 50px;'>Condense and Conquer📌</h1>",
        unsafe_allow_html=True,
    )

    Analytics_intro = "<p style='font-size: 22px; text-align: left;'>This module is designed to streamline information retrieval during placement preparations or general inquiries by providing concise summaries of specific topics. Users can provide the URL of a website containing the relevant blog post, and the module will automatically scrape the content and generate a summary. This eliminates the need to review extensive content, saving time and effort. However, please note that content behind paywalls cannot be processed due to access restrictions. In such cases, users may need to provide an alternative source or a different post. </p>"
    st.markdown(Analytics_intro, unsafe_allow_html=True)
    st.markdown("***")


    summarized_text = ""
    original_text = ""
    llm_model,text_splitter_semantic = laoding_models()
    summary_chain = get_summary_chain(llm_model)


    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        url = st.text_input("Enter the blog post URL")
        bool_summarized_content = False
        summarize_col, registered_col = st.columns(2, gap="large")
        summarize_bt = st.button("Summarize the content", use_container_width=True)
        if summarize_bt:
            if not url:
                st.error("Please enter a URL first.")
            else:
                try:
                    ARTICLE = extract_process(url)
                    original_text = ARTICLE
                    docs = text_splitter_semantic.create_documents([original_text])
                    summary = summary_chain.run(docs)
                    summarized_text = summary
                  
                    bool_summarized_content = True
                    st.markdown("<h3>Summarized content️️</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 20px;'>{summarized_text}</p>", unsafe_allow_html=True)
                    
                    
                    st.write("")

                except Exception as e:
                    st.error(
                        f"Error summarizing the content due to wrong url or paywall: {e}"
                    )

    with col2:
        st.markdown("<h3>Unveiling the Magic 🪄</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='background-color: #C3E8FF; padding: 20px; border-radius: 10px; font-size: 20px;'>URLs submitted undergo HTTPS verification; if successful and devoid of paywalls, the article's content is extracted. A model generates variable chunks of summarized text for efficient data loading. These summarized chunks are stored in a text file for user access..</p>",
            unsafe_allow_html=True,
        )

        st.link_button(
            "Original Blog Post (Anti Scaraping disabled)",
            "https://ai.meta.com/blog/large-language-model-llama-meta-ai/",
            use_container_width=True,
        )

        wordcnt_col1, wordcnt_col2 = st.columns(2, gap="large")
        if bool_summarized_content:
            original_wc = llm_model.get_num_tokens(original_text)
            summary_wc = llm_model.get_num_tokens(summarized_text)

            row = st.columns(2)
            index = 0
            for col in row:
                tile = col.container(height=150)  # Adjust the height as needed
                if index == 0:
                    tile.metric(
                        label="Original Token Count",
                        value=original_wc,
                        delta="Complete content : 100%",
                    )
                else:
                    tile.metric(
                        label="Summary Token Count",
                        value=summary_wc,
                        delta="Condense % : "
                        + str(int(((original_wc - summary_wc) / original_wc) * 100)),
                    )
                index = index + 1

            download_summary(summary)


text_summarization_page()
