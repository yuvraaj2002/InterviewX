import streamlit as st
from typing import Tuple
import streamlit as st
from src.Likedin_agent import lookup as linkedin_lookup_agent
from src.Scrapte_linkedin_data import scrape_linkedin_profile
from src.Chains import (
    get_summary_chain,
    get_interests_chain,
    get_ice_breaker_chain,
)
from src.Parsers import (
    summary_parser,
    topics_of_interest_parser,
    ice_breaker_parser,
    Summary,
    IceBreaker,
    TopicOfInterest,
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.0rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def ice_break_with(name: str) -> Tuple[Summary, IceBreaker, TopicOfInterest, str]:
    try:
        linkedin_username = linkedin_lookup_agent(name=name)
        linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

        summary_chain = get_summary_chain()
        summary_and_facts = summary_chain.run(information=linkedin_data)
        summary_and_facts = summary_parser.parse(summary_and_facts)

        interests_chain = get_interests_chain()
        interests = interests_chain.run(information=linkedin_data)
        interests = topics_of_interest_parser.parse(interests)

        ice_breaker_chain = get_ice_breaker_chain()
        ice_breakers = ice_breaker_chain.run(information=linkedin_data)
        ice_breakers = ice_breaker_parser.parse(ice_breakers)

        return (
            summary_and_facts,
            interests,
            ice_breakers,
            linkedin_data.get("profile_pic_url"),
        )

    except :
        # Handle the StopCandidateException here
        st.error("An error occurred: StopCandidateException")


def Ice_breaker_page():
    st.title("ICE breaker👋")
    st.markdown(
        "<p style='font-size: 20px; text-align: left;padding-right: 2rem;padding-bottom: 1rem;'>In preparation for an interview, it is advantageous to acquaint oneself with the background and interests of the interviewee. This facilitates effective communication and rapport-building during the interview process. To streamline this preparatory phase, a module has been developed to provide concise summaries, pertinent facts, and tailored icebreakers for the interviewee. By inputting the interviewee's name or relevant details, users can access a comprehensive overview to enhance their interaction and engagement during the interview.</p>",
        unsafe_allow_html=True,
    )
    input_col, details_col = st.columns(spec=(0.8, 2), gap="large")

    # Initialize content with default values
    content = {"summary_and_facts": None, "interests": None, "ice_breakers": None, "profile_pic_url": None}
    content_extracted = False

    with input_col:
        name = st.text_input("Enter the name of person")
        info_bt = st.button("Extract information⛏️", use_container_width=True)
        if name:
            if info_bt:

                summary_and_facts, interests, ice_breakers, profile_pic_url = (
                    ice_break_with(name)
                )
                content = {
                    "summary_and_facts": summary_and_facts.to_dict(),
                    "interests": interests.to_dict(),
                    "ice_breakers": ice_breakers.to_dict(),
                    "profile_pic_url": profile_pic_url,
                }
                content_extracted = True
        else:
            st.error("Please provide a name to see get details")

    with input_col:
        if content['profile_pic_url'] is not None and content_extracted:
            image_add = "https://media.licdn.com/dms/image/C4D03AQGlv35ItbkHBw/profile-displayphoto-shrink_400_400/0/1610187870291?e=1716422400&v=beta&t=Hb9E0lFt9Ivhz7JhpoZ4h-Jf0KMDpncL21GJ2wrQCT8"
            st.markdown(f'<img src="{image_add}" style="border-radius: 50%;">', unsafe_allow_html=True)
            #st.image(content['profile_pic_url'])

    with details_col:
        if name and content_extracted:
            st.title("Meet " + name)

            summary = content['summary_and_facts']["summary"]
            if summary is not None:
                st.markdown(
                    f"<p style='text-align: left; font-size: 20px; '>{summary}</p>",
                    unsafe_allow_html=True,
                )
            facts = content['summary_and_facts']["facts"]
            if facts is not None:
                for fact in facts:
                    st.markdown(
                        f"<p style='text-align: left; font-size: 20px;'>➡️ {fact}</p>",
                        unsafe_allow_html=True,
                    )

            st.markdown(
                "<h4 style='text-align: left;'>Break the Ice with,</h4>",
                unsafe_allow_html=True,
            )
            ice_breakers = content['ice_breakers']["ice_breakers"]
            if ice_breakers is not None:
                for ice_breaker in ice_breakers:
                    st.markdown(
                        f"<p style='text-align: left; font-size: 20px;background-color:#C3E8FF;padding:0.5rem;'>{ice_breaker}</p>",
                        unsafe_allow_html=True,
                    )



Ice_breaker_page()
