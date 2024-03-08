import os
import requests
import streamlit as st


def get_process_linkedin_data(linkedin_profile_url):
    """
    Fetches and processes LinkedIn data using ProxyCurl API.

    Args:
        linkedin_profile_url (str): URL of the LinkedIn profile.

    Returns:
        dict: Processed LinkedIn data.

    """

    api_key = os.environ.get("PROXYCURL_API_KEY")
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"url": linkedin_profile_url}

    # response = requests.get(api_endpoint, params=params, headers=headers)
    response = requests.get(
        "https://gist.githubusercontent.com/paul-villalobos/c59167135dea9081021d53b2faebad66/raw/88ec8ce557de26be827cd29a617dceee3eb9306d/eden-marco.json"
    )
    data = response.json()

    cleaned_data = {
        key: value
        for key, value in data.items()
        if value not in ([], "", None) and key != "people_also_viewed"
    }

    if cleaned_data.get("groups"):
        for group_dict in cleaned_data.get("groups"):
            group_dict.pop("profile_pic_url")
    return cleaned_data


def Ice_breaker_page():
    st.title("Ice breaker")
    linkedin_data = get_process_linkedin_data("somehignsg")
    st.write(linkedin_data)
