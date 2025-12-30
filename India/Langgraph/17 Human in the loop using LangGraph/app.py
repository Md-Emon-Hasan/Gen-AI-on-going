import streamlit as st
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import uuid
import time

load_dotenv()

# **************************************** Configuration *************************

st.set_page_config(
    page_title="HITL Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .approval-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .tool-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .danger-badge {
        background: #ff4757;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .safe-badge {
        background: #26de81;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# **************************************** LLM & Tools *************************

@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3,
    )

llm = initialize_llm()

# Tools
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    time.sleep(1)  # Simulate API call
    return f"Email sent to {to}\nSubject: {subject}"

@tool
def delete_file(filepath: str) -> str:
    """Delete a file (DANGEROUS)"""
    time.sleep(1)
    return f"Deleted: {filepath}"

@tool
def create_task(title: str, priority: str = "medium") -> str:
    """Create a task"""
    return f"Task: {title} (Priority: {priority})"

@tool
def search_web(query: str) -> str:
    """Search the web"""
    return f"Found results for: {query}"

DANGEROUS_TOOLS = ["delete_file", "send_email"]
tools = [send_email, delete_file, create_task, search_web]

# **************************************** Graph Setup *************************

class State(TypedDict):
    messages: Annotated[list, add_messages]
    pending_tools: list
    approved: bool

def agent(state: State):
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state['messages'])
    
    pending = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        pending = response.tool_calls
    
    return {"messages": [response], "pending_tools": pending}

def execute_tools(state: State):
    results = []
    
    for tc in state.get('pending_tools', []):
        tool_name = tc.get('name')
        tool_args = tc.get('args', {})
        tool_id = tc.get('id', 'unknown')
        
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                results.append(ToolMessage(content=result, tool_call_id=tool_id))
                break
    
    return {"messages": results}

def needs_approval(state: State):
    pending = state.get('pending_tools', [])
    
    for tc in pending:
        if tc.get('name') in DANGEROUS_TOOLS:
            return "approval_needed"
    
    if pending:
        return "execute"
    
    return END

@st.cache_resource
def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("execute", execute_tools)
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", needs_approval, {
        "approval_needed": END,
        "execute": "execute",
        END: END
    })
    builder.add_edge("execute", "agent")
    
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["execute"]
    )

graph = build_graph()

# **************************************** Session State *************************

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pending' not in st.session_state:
    st.session_state.pending = None

# **************************************** Helper Functions *************************

def get_state():
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        return graph.get_state(config)
    except:
        return None

def approve_and_execute():
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Continue execution
    list(graph.stream(None, config=config))
    
    st.session_state.pending = None
    
    # Get results
    state = get_state()
    if state:
        last_msg = state.values['messages'][-1]
        result = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        st.session_state.messages.append({'role': 'assistant', 'content': result})

def reject():
    st.session_state.pending = None
    st.session_state.messages.append({
        'role': 'assistant',
        'content': 'Action cancelled by user'
    })

# **************************************** Sidebar *************************

with st.sidebar:
    st.title("HITL Assistant")
    st.markdown("Human-in-the-Loop AI")
    
    st.markdown("---")
    
    # Stats
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Session", st.session_state.thread_id[:8] + "...")
    
    st.markdown("---")
    
    # Tools info
    st.subheader("Available Tools")
    
    for tool in tools:
        is_dangerous = tool.name in DANGEROUS_TOOLS
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{tool.name}**")
                st.caption(tool.description)
            with col2:
                if is_dangerous:
                    st.markdown('<span class="danger-badge">⚠️ DANGER</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="safe-badge">✓ SAFE</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

# **************************************** Main Area *************************

st.title("Chat with HITL Assistant")

# Check for pending approvals
state = get_state()
if state and state.values.get('pending_tools') and not st.session_state.pending:
    pending = state.values['pending_tools']
    dangerous = [t for t in pending if t.get('name') in DANGEROUS_TOOLS]
    if dangerous:
        st.session_state.pending = dangerous

# Show approval card
if st.session_state.pending:
    st.markdown("""
    <div class="approval-card">
        <h3>Approval Required</h3>
        <p>The AI wants to perform dangerous actions. Please review and approve.</p>
    </div>
    """, unsafe_allow_html=True)
    
    for tool_call in st.session_state.pending:
        tool_name = tool_call.get('name', 'Unknown')
        tool_args = tool_call.get('args', {})
        
        st.markdown(f"""
        <div class="tool-card">
            <h4>{tool_name.replace('_', ' ').title()}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.json(tool_args)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Approve & Execute", use_container_width=True, type="primary"):
            with st.spinner("Executing..."):
                approve_and_execute()
            st.success("Executed successfully!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("Reject", use_container_width=True):
            reject()
            st.error("Action cancelled")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Process
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            list(graph.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            ))
            
            state = get_state()
            
            if state.values.get('pending_tools'):
                pending = state.values['pending_tools']
                dangerous = [t for t in pending if t.get('name') in DANGEROUS_TOOLS]
                
                if dangerous:
                    st.session_state.pending = dangerous
                    response = "**Waiting for approval...**"
                else:
                    response = "Executing safe tools..."
            else:
                last_msg = state.values['messages'][-1]
                response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            
            st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
    
    st.rerun()

# Quick examples
st.markdown("---")
st.markdown("### Try These Examples")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Send Email", use_container_width=True):
        prompt = "Send an email to boss@company.com about the project"
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.rerun()

with col2:
    if st.button("Delete File", use_container_width=True):
        prompt = "Delete the file old_data.csv"
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.rerun()

with col3:
    if st.button("Create Task", use_container_width=True):
        prompt = "Create a task to review the code"
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.rerun()