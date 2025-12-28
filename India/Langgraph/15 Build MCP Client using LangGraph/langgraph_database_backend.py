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
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
)

# Initialize Regular Tools
tavily_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
)

# **************************************** MCP Tools Setup *************************

# Global MCP session
mcp_session = None
mcp_tools_list = []

async def initialize_mcp():
    """Initialize MCP connection"""
    global mcp_session, mcp_tools_list
    
    try:
        # Example: Connect to filesystem MCP server
        # Replace with your MCP server command
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            env=None
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                response = await session.list_tools()
                mcp_session = session
                
                # Convert MCP tools to LangChain tools
                for mcp_tool in response.tools:
                    mcp_tools_list.append(create_langchain_tool_from_mcp(mcp_tool))
                
                print(f"✅ MCP initialized with {len(mcp_tools_list)} tools")
                return session
                
    except Exception as e:
        print(f"⚠️ MCP initialization failed: {e}")
        print("Continuing without MCP tools...")
        return None

def create_langchain_tool_from_mcp(mcp_tool):
    """Convert MCP tool to LangChain tool"""
    
    @tool
    def mcp_tool_wrapper(query: str) -> str:
        f"""
        {mcp_tool.description}
        
        Args:
            query: The input query for {mcp_tool.name}
        """
        try:
            # Call MCP tool
            result = asyncio.run(call_mcp_tool(mcp_tool.name, {"query": query}))
            return str(result)
        except Exception as e:
            return f"Error calling {mcp_tool.name}: {str(e)}"
    
    # Set tool name and description
    mcp_tool_wrapper.name = mcp_tool.name
    mcp_tool_wrapper.description = mcp_tool.description
    
    return mcp_tool_wrapper

async def call_mcp_tool(tool_name: str, arguments: dict):
    """Call an MCP tool"""
    global mcp_session
    
    if not mcp_session:
        return "MCP session not initialized"
    
    try:
        result = await mcp_session.call_tool(tool_name, arguments)
        return result.content
    except Exception as e:
        return f"Error: {str(e)}"

# Custom MCP-like tools (fallback if MCP server not available)
@tool
def read_file(filepath: str) -> str:
    """
    Read contents of a file from the filesystem.
    
    Args:
        filepath: Path to the file to read
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File content from {filepath}:\n\n{content[:1000]}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def list_directory(dirpath: str = ".") -> str:
    """
    List files and directories in a given path.
    
    Args:
        dirpath: Path to the directory to list (default: current directory)
    """
    try:
        import os
        items = os.listdir(dirpath)
        files = [f for f in items if os.path.isfile(os.path.join(dirpath, f))]
        dirs = [d for d in items if os.path.isdir(os.path.join(dirpath, d))]
        
        result = f"Directory: {dirpath}\n\n"
        result += f"📁 Directories ({len(dirs)}):\n"
        result += "\n".join([f"  - {d}/" for d in dirs[:10]])
        result += f"\n\n📄 Files ({len(files)}):\n"
        result += "\n".join([f"  - {f}" for f in files[:10]])
        
        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Combine all tools
tools = [tavily_tool, wikipedia, read_file, list_directory, get_current_time]

# Try to initialize MCP (async)
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Add MCP tools if available
if mcp_tools_list:
    tools.extend(mcp_tools_list)

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

def get_available_tools():
    """Get list of available tools with descriptions"""
    tool_info = []
    for tool in tools:
        tool_info.append({
            'name': tool.name,
            'description': tool.description
        })
    return tool_info