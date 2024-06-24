import streamlit as st
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import time

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


@st.cache_resource
def summarizing_pipeline():

    # Instantiating the summarization pipeline using t5-base model
    summarizer = pipeline(
        "summarization",
        model="t5-base",
        tokenizer="t5-base",
        framework="pt",
    )
    return summarizer


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


def chunk_creation(article, max_chunk=512):
    """
    Chunks an article into sentences, respecting sentence boundaries and a maximum chunk size.

    Args:
        article: The text of the article to be chunked.
        max_chunk: The maximum number of tokens allowed in a chunk.

    Returns:
        A list of chunks, each represented as a string.
    """
    article = article.replace(".", ".<eos>")
    article = article.replace("?", "?<eos>")
    article = article.replace("!", "!<eos>")

    # Split the article into sentences based on the '<eos>' marker.
    sentences = article.split("<eos>")

    # Initialize variables for chunk creation.
    current_chunk = 0
    chunks = []

    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if (
                len(chunks[current_chunk].split(" ")) + len(sentence.split(" "))
                <= max_chunk
            ):
                chunks[current_chunk] += " " + sentence
            else:
                current_chunk += 1
                chunks.append(sentence)
        else:
            chunks.append(sentence)

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = chunks[chunk_id].strip()

    return chunks


def text_summarization_page():
    st.markdown(
        "<h1 style='text-align: center; font-size: 50px;'>Condense and Conquer📌</h1>",
        unsafe_allow_html=True,
    )

    Analytics_intro = "<p style='font-size: 22px; text-align: center;'>This module is designed to streamline information retrieval during placement preparations or general inquiries by providing concise summaries of specific topics. Users can provide the URL of a website containing the relevant blog post, and the module will automatically scrape the content and generate a summary. This eliminates the need to review extensive content, saving time and effort. However, please note that content behind paywalls cannot be processed due to access restrictions. In such cases, users may need to provide an alternative source or a different post. </p>"
    st.markdown(Analytics_intro, unsafe_allow_html=True)
    summarizer = summarizing_pipeline()
    st.markdown("***")

    summary = ""
    original_text = ""
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
                    CHUNKS = chunk_creation(ARTICLE)
                    results = summarizer(
                        CHUNKS, max_length=150, min_length=50, do_sample=False
                    )
                    summarized_text = " ".join(
                        [summ["summary_text"] for summ in results]
                    )
                    bool_summarized_content = True
                    summary = summarized_text
                    st.markdown("<h3>Summarized content️️</h3>", unsafe_allow_html=True)
                    markdown_text = f"<p style='font-size: 20px;'>{summarized_text}</p>"
                    st.markdown(markdown_text, unsafe_allow_html=True)
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
            "https://www.codecontent.net/post/introduction-to-llama",
            use_container_width=True,
        )

        wordcnt_col1, wordcnt_col2 = st.columns(2, gap="large")
        if bool_summarized_content:
            original_wc = len(original_text.split())
            summary_wc = len(summary.split())

            row = st.columns(2)
            index = 0
            for col in row:
                tile = col.container(height=150)  # Adjust the height as needed
                if index == 0:
                    tile.metric(
                        label="Original Word Count",
                        value=original_wc,
                        delta="Complete content : 100%",
                    )
                else:
                    tile.metric(
                        label="Summary Word Count",
                        value=summary_wc,
                        delta="Condense % : "
                        + str(int(((original_wc - summary_wc) / original_wc) * 100)),
                    )
                index = index + 1

            download_summary(summary)


text_summarization_page()
