from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
)

# Initialize Tools
tavily_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
)

tools = [tavily_tool, wikipedia]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def should_continue(state: ChatState):
    """Determine whether to continue to tools or end"""
    messages = state['messages']
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

def chat_node(state: ChatState):
    """Main chat node with tool support"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def format_tool_results(state: ChatState):
    """Format tool results with references"""
    messages = state['messages']
    last_message = messages[-1]
    
    if hasattr(last_message, 'content'):
        # Check if this is a tool response
        tool_messages = [msg for msg in messages if hasattr(msg, 'name')]
        if tool_messages:
            # Create a summary with references
            formatted_response = llm.invoke([
                HumanMessage(content=f"""Based on the following tool results, provide a comprehensive answer with references.

Tool Results:
{last_message.content}

Format your response naturally and mention which sources you used (Tavily Search or Wikipedia).""")
            ])
            return {"messages": [formatted_response]}
    
    return {"messages": []}

# Database setup
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

# Add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge("tools", "chat_node")

# Compile
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    """Retrieve all conversation threads"""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)