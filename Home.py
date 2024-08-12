import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="InterviewX",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .top-margin{
                    margin-top: 4rem;
                    margin-bottom:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


# Main page function
def main_page():
    Overview_col, Img_col = st.columns(spec=(1.3, 1), gap="large")

    with Overview_col:
        # Content for main page
        st.markdown(
            "<h1 style='text-align: left; font-size: 85px; '>InterviewX</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 22px; text-align: left;'>Are you ready to ace your next interview with confidence? Welcome to InterviewX, your ultimate companion on the journey to interview success. Harnessing the power of cutting-edge AI technology, InterviewX offers a comprehensive suite of modules designed to empower individuals at every stage of their interview preparation.</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div>
            <ul>
                <li><p style='font-size: 22px; text-align: left;'><strong>Avoid scams effortlessly!</strong> InterviewX helps you distinguish genuine job postings from fraudulent ones, saving you time and effort.</p></li>
                <li><p style='font-size: 22px; text-align: left;'><strong>Resume Content Matching and Top 5 Questions:</strong> Craft the perfect resume! InterviewX analyzes your resume and matches it with job descriptions, revealing the top five expected questions based on your qualifications.</p></li>
                <li><p style='font-size: 22px; text-align: left;'><strong>Real-Time Posture Analysis:</strong> Project confidence flawlessly! InterviewX's posture analysis gives instant feedback to refine your body language for a professional impression during interviews.</p></li>
                <li><p style='font-size: 22px; text-align: left;'><strong>Text Summarization:</strong> Stay informed, save time! InterviewX summarizes lengthy materials quickly and efficiently. Just input the blog post link for concise insights.</p></li>
                <li><p style='font-size: 22px; text-align: left;'><strong>OmniChat:</strong> Effortlessly enhance your media experience! Upload a YouTube URL or PDF, and our AI will create searchable transcripts and text extractions. Access detailed insights instantly with our advanced AI technology.</p></li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with Img_col:
        st.markdown("<div class='top-margin'> </div>", unsafe_allow_html=True)
        st.image("artifacts/Main_Page.jpg")
        st.write("")

        social_col1, social_col2, social_col3, social_col4 = st.columns(
            spec=(1, 1, 1, 1), gap="large"
        )
        with social_col1:
            st.link_button(
                "Github👨‍💻",
                use_container_width=True,
                url="https://github.com/yuvraaj2002",
            )

        with social_col2:
            st.link_button(
                "Linkedin🧑‍💼",
                use_container_width=True,
                url="https://www.linkedin.com/in/yuvraj-singh-a4430a215/",
            )

        with social_col3:
            st.link_button(
                "Twitter🧠",
                use_container_width=True,
                url="https://twitter.com/Singh_yuvraaj1",
            )

        with social_col4:
            st.link_button(
                "Blogs✒️", use_container_width=True, url="https://yuvraj01.hashnode.dev/"
            )


main_page()
