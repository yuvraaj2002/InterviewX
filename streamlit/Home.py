import streamlit as st

st.set_page_config(page_title="AI_Tutor", page_icon="🧑‍🏫", layout="wide")
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


def main_page():
    # Set the background image with a darker gradient overlay
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: 100vw 100vh;  /* This sets the size to cover 100% of the viewport width and height */
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """

    # Apply the background image with a darker gradient overlay
    st.markdown(background_image, unsafe_allow_html=True)

    with st.container():
        # Title and the introduction text
        st.markdown(
            "<h1 style='text-align: center; font-size: 80px; padding-top: 8rem;'>FindHome.AI</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 25px; text-align: center;padding-left: 2rem;padding-right: 2rem;'>Discover your dream home effortlessly with our AI-Powered Home Finder in Gurgaon. Using advanced algorithms, it streamlines house-hunting, offering personalized recommendations. Integrated with a cutting-edge loan eligibility module, it provides real-time insights for informed financial decisions. This powerful feature considers factors like credit score, ensuring your dream home is within budget. Revolutionize your homebuying experience with the future of technology, combining intelligent home search and personalized financial guidance.</p>",
            unsafe_allow_html=True,
        )


main_page()
