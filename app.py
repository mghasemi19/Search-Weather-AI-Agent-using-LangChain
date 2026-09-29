import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults


# =========================================================
# ENVIRONMENT
# =========================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + OpenWeatherMap AI Agent using LangChain 1.x")


# =========================================================
# API KEY CHECKS
# =========================================================

missing_keys = []

if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not OPENWEATHER_API_KEY:
    missing_keys.append("OPENWEATHER_API_KEY")

if missing_keys:
    st.error("Missing environment variables: " + ", ".join(missing_keys))
    st.stop()


# =========================================================
# SEARCH TOOL
# =========================================================

search_tool = TavilySearchResults(max_results=2)


# =========================================================
# WEATHER TOOL
# =========================================================

@tool
def get_weather_data(city: str) -> str:
    """Fetch the current weather for a city using OpenWeatherMap."""

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        return f"Weather request failed for {city}: {e}"

    if "main" not in data or "weather" not in data:
        return f"Could not fetch weather data for {city}"

    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    resolved_city = data.get("name", city)

    return (
        f"City: {resolved_city}\n"
        f"Temperature: {temperature}°C\n"
        f"Feels like: {feels_like}°C\n"
        f"Weather: {description}\n"
        f"Humidity: {humidity}%"
    )


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)


# =========================================================
# AGENT
# =========================================================

tools = [
    search_tool,
    get_weather_data,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful research assistant. "
        "Use available tools whenever necessary. "
        "For current factual information, use web search rather than guessing. "
        "For current weather, always use the weather tool."
    ),
)


# =========================================================
# USER INPUT
# =========================================================

user_query = st.text_input(
    "Enter your query:",
    placeholder=(
        "Example: Find the capital of India "
        "and tell me its current weather."
    ),
)


# =========================================================
# RUN AGENT
# =========================================================

if st.button("Run Agent", type="primary"):

    if not user_query.strip():
        st.warning("Please enter a query.")

    else:
        with st.spinner("Agent is working..."):

            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query,
                            }
                        ]
                    }
                )

                final_message = response["messages"][-1]

                st.success("Response generated")

                st.markdown("## Final Response")
                st.write(final_message.content)

                with st.expander("Show agent messages"):
                    for i, message in enumerate(response["messages"]):
                        st.markdown(f"### Message {i}")
                        st.write(f"Type: {type(message).__name__}")
                        st.write(message)

            except Exception as e:
                st.error(f"Error while running agent: {e}")
