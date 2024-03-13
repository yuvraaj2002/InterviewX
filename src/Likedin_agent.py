from dotenv import load_dotenv
import streamlit as st
load_dotenv()
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
from langchain import hub
from langchain_core.prompts import PromptTemplate
from Tools.scrape_tool import get_profile_url
from langchain_google_genai import ChatGoogleGenerativeAI


@st.cache_resource
def load_model():
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key="AIzaSyBmdKcpl5PTuVBVsHsOdAlotEOOAInFxoU"
    )
    return llm


def lookup(name: str) -> str:
    llm = load_model()
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url
