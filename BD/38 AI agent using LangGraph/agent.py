import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Load API keys
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Set them in the environment (LangChain uses this internally)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Use Gemini Pro via LangChain wrapper
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

# Tool: Tavily for web search
search = TavilySearchResults(max_results=3)

# Combine tools into a REAct Agent
tools = [search]
agent_executor = create_react_agent(chat_model, tools)

# Invoke the agent
response = agent_executor.invoke({
    "messages": [HumanMessage(content="Tell me the recent movies list in 2025")]
})

# Print the output
for msg in response['messages']:
    print(msg.content)