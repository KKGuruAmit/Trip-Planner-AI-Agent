# Import Required Library
import os
import datetime
import streamlit as st
from crewai import Agent
from crewai import Task
from langchain_groq import ChatGroq
from crewai import Crew,Process

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

# LLM Monitering
os.environ['LANGCHAIN_API_KEY']="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"
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Trip Planner AI Agent Moniter"

st.subheader("**Multi AI Agent Trip Planner...**")

# Getting Task From Web
with st.form(key='Query',clear_on_submit=True):
    location=st.text_input(label='**From where will you be Traveling from?**')
    cities=st.text_input(label='**What are the Cities options you are Interested in Visiting?**')
    date_range=str(st.date_input(label='**What is the Date you are Interested in Traveling?**',
                             value=datetime.date(2024,9,4),format='DD/MM/YYYY'))
    interest=st.text_input(label='**What are some of your High level Interests and Hobbies?**')
    submit_button = st.form_submit_button('Submit.')
    if submit_button:
        st.info('Input Details...')
        st.markdown(f'Travel Location: {location} ...')
        st.markdown(f'Cities Name: {cities} ...')
        st.markdown(f'Traveling Date: {date_range} ...')
        st.markdown(f'Interests and Hobbies Name: {interest} ...')

# Creating LLM Variable
os.environ['GROQ_API_KEY']='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'
LLM_Model=ChatGroq(model='llama3-70b-8192',api_key=os.getenv('GROQ_API_KEY'))

# Creating Agents
city_selection_agent = Agent(
    role='City Selection Expert',
    goal='Select the best city based on weather, season, and prices',
    backstory='An expert in analyzing travel data to pick ideal destinations',
    verbose=True,
    allow_delegation=False,
    tools=[SearchTools.search_internet,
           BrowserTools.scrape_and_summarize_website,],
    llm=LLM_Model
)

local_expert_agent = Agent(
    role='Local Expert at this city',
    goal='Provide the BEST insights about the selected city',
    backstory="""A knowledgeable local guide with extensive 
    information about the city, it's attractions and customs""",
    verbose=True,
    allow_delegation=False,
    tools=[SearchTools.search_internet,
           BrowserTools.scrape_and_summarize_website,],
    llm=LLM_Model
)

travel_concierge_agent = Agent(
    role='Amazing Travel Concierge',
    goal="""Create the most amazing travel itineraries with 
    budget and packing suggestions for the city""",
    backstory="""Specialist in travel planning and 
    logistics with decades of experience""",
    verbose=True,
    tooles=[SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,],
    llm=LLM_Model
)

# Creating Task for Agent
identify_task = Task(
    description='''Analyze and select the best city for the trip based 
    on specific criteria such as weather patterns, seasonal
    events, and travel costs. This task involves comparing
    multiple cities, considering factors like current weather
    conditions, upcoming cultural or seasonal events, and
    overall travel expenses. 
                
    Your final answer must be a detailed
    report on the chosen city, and everything you found out
    about it, including the actual flight costs, weather 
    forecast and attractions.
                
    Traveling from: {origin}
    City Options: {cities}
    Trip Date: {range}
    Traveler Interests: {interests}''',
    expected_output='''Detailed report on the chosen city 
    including flight costs, weather forecast, and attractions''',
    agent=city_selection_agent
)

gather_task = Task(
    description='''As a local expert on this city you must compile an 
    in-depth guide for someone traveling there and wanting 
    to have THE BEST trip ever!
    Gather information about key attractions, local customs,
    special events, and daily activity recommendations.
    Find the best spots to go to, the kind of place only a
    local would know.
    This guide should provide a thorough overview of what 
    the city has to offer, including hidden gems, cultural
    hotspots, must-visit landmarks, weather forecasts, and
    high level costs.
                
    The final answer must be a comprehensive city guide, 
    rich in cultural insights and practical tips, 
    tailored to enhance the travel experience.

    Trip Date: {range}
    Traveling from: {origin}
    Traveler Interests: {interests}''',
    expected_output='''Comprehensive city guide including hidden gems, 
    cultural hotspots, and practical travel tips''',
    agent=local_expert_agent
)
plan_task = Task(
    description='''Expand this guide into a full 7-day travel 
    itinerary with detailed per-day plans, including 
    weather forecasts, places to eat, packing suggestions, 
    and a budget breakdown.
                
    You MUST suggest actual places to visit, actual hotels 
    to stay and actual restaurants to go to.
                
    This itinerary should cover all aspects of the trip, 
    from arrival to departure, integrating the city guide
    information with practical travel logistics.
                
    Your final answer MUST be a complete expanded travel plan,
    formatted as markdown, encompassing a daily schedule,
    anticipated weather conditions, recommended clothing and
    items to pack, and a detailed budget, ensuring THE BEST
    TRIP EVER. Be specific and give it a reason why you picked
    each place, what makes them special!

    Trip Date: {range}
    Traveling from: {origin}
    Traveler Interests: {interests}''',
    expected_output='''Complete expanded travel plan with daily schedule, 
    weather conditions, packing suggestions, and budget breakdown''',
    agent=travel_concierge_agent
)

inputs={
    'origin':location,
    'cities':cities,
    'range':date_range,
    'interests':interest,
}

crew=Crew(
    agents=[city_selection_agent,local_expert_agent,travel_concierge_agent],
    tasks=[identify_task,gather_task,plan_task],
    verbose=True,
    process=Process.sequential,
    manager_llm=LLM_Model
)

if st.button('Generate'):
    with st.spinner('Generate Response...'):
        result=crew.kickoff(inputs=inputs)
        res=str(result)
        st.info('Here is Response')
        st.markdown(result)
        st.download_button(label='Download Text File',
                           file_name=f'{cities} Trip Plan.txt',data=res)