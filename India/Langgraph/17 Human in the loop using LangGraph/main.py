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

load_dotenv()

# Page config
st.set_page_config(
    page_title="HITL Assistant",
    layout="wide"
)

# **************************************** LLM & Tools Setup *************************

@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3,
    )

llm = initialize_llm()

# Define tools
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone"""
    return f"Email sent to {to}\nSubject: {subject}\nBody: {body}"

@tool
def delete_file(filepath: str) -> str:
    """Delete a file (DANGEROUS operation)"""
    return f"File deleted: {filepath}"

@tool
def create_task(title: str, priority: str = "medium") -> str:
    """Create a task in the system"""
    return f"Task created: {title} (Priority: {priority})"

@tool
def book_meeting(attendees: str, date: str, duration: str = "30min") -> str:
    """Book a meeting with attendees"""
    return f"Meeting booked with {attendees} on {date} for {duration}"

DANGEROUS_TOOLS = ["delete_file", "send_email"]
tools = [send_email, delete_file, create_task, book_meeting]
llm_with_tools = llm.bind_tools(tools)

# **************************************** Graph States *************************

class ToolHITLState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_tool_calls: list
    approved_tools: list
    human_decision: str
    iteration_count: int

# **************************************** Graph Nodes *************************

def agent_node(state: ToolHITLState):
    """AI decides what to do"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    
    pending_tools = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        pending_tools = response.tool_calls
    
    return {
        "messages": [response],
        "pending_tool_calls": pending_tools,
        "iteration_count": state.get('iteration_count', 0) + 1
    }

def tool_execution_node(state: ToolHITLState):
    """Execute approved tools"""
    approved_tools = state.get('approved_tools', [])
    
    if not approved_tools:
        return {"messages": [AIMessage(content="No tools were approved for execution.")]}
    
    tool_results = []
    
    for tool_call in approved_tools:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_id = tool_call.get('id', 'unknown')
        
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    tool_results.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        )
                    )
                except Exception as e:
                    tool_results.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_id
                        )
                    )
                break
    
    return {"messages": tool_results}

def should_continue_tools(state: ToolHITLState):
    """Check if we need human approval"""
    pending = state.get('pending_tool_calls', [])
    
    if not pending:
        return END
    
    # Check if any dangerous tools
    for tool_call in pending:
        if tool_call.get('name') in DANGEROUS_TOOLS:
            return "wait_approval"
    
    # Auto-approve safe tools
    return "execute"

def should_execute(state: ToolHITLState):
    """Check human decision"""
    decision = state.get('human_decision', '')
    
    if decision == 'approved':
        return "execute"
    elif decision == 'rejected':
        return END
    else:
        return "wait_approval"

# **************************************** Build Graph *************************

@st.cache_resource
def build_hitl_graph():
    graph_builder = StateGraph(ToolHITLState)
    
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("execute", tool_execution_node)
    
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue_tools,
        {
            "wait_approval": END,  # Pause for approval
            "execute": "execute",
            END: END
        }
    )
    graph_builder.add_edge("execute", "agent")
    
    checkpointer = MemorySaver()
    return graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"]
    )

hitl_graph = build_hitl_graph()

# **************************************** Session State *************************

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pending_approval' not in st.session_state:
    st.session_state.pending_approval = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# **************************************** Helper Functions *************************

def get_tool_emoji(tool_name):
    emojis = {
        'send_email': '📧',
        'delete_file': '🗑️',
        'create_task': '✅',
        'book_meeting': '📅',
    }
    return emojis.get(tool_name, '🔧')

def reset_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_approval = None
    st.session_state.conversation_history = []

def get_graph_state():
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        state = hitl_graph.get_state(config)
        return state
    except:
        return None

# **************************************** UI Layout *************************

# Header
st.title("Human-in-the-Loop Assistant")
st.markdown("*AI assistant that asks for your approval before taking critical actions*")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Mode selection
    mode = st.radio(
        "HITL Mode",
        ["Strict (Approve all dangerous tools)", 
         "Auto (Auto-approve safe tools)"],
        index=1
    )
    
    st.markdown("---")
    
    # Dangerous tools list
    st.subheader("Dangerous Tools")
    st.caption("These tools require human approval:")
    for tool_name in DANGEROUS_TOOLS:
        emoji = get_tool_emoji(tool_name)
        st.markdown(f"{emoji} **{tool_name.replace('_', ' ').title()}**")
    
    st.markdown("---")
    
    # Available tools
    with st.expander("All Available Tools"):
        for tool in tools:
            emoji = get_tool_emoji(tool.name)
            st.markdown(f"**{emoji} {tool.name}**")
            st.caption(tool.description)
    
    st.markdown("---")
    
    # Reset button
    if st.button("New Conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    
    # Stats
    st.markdown("---")
    st.subheader("Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Thread ID", st.session_state.thread_id[:8] + "...")

# **************************************** Main Chat Area *************************

# Check for pending approvals
state = get_graph_state()

if state and state.values.get('pending_tool_calls'):
    pending_tools = state.values.get('pending_tool_calls', [])
    
    # Check if any dangerous tools need approval
    dangerous_pending = [t for t in pending_tools if t.get('name') in DANGEROUS_TOOLS]
    
    if dangerous_pending and not st.session_state.pending_approval:
        st.session_state.pending_approval = dangerous_pending

# Display pending approval request
if st.session_state.pending_approval:
    st.warning("**Approval Required**")
    
    with st.container():
        st.markdown("### Tool Execution Request")
        
        for i, tool_call in enumerate(st.session_state.pending_approval, 1):
            tool_name = tool_call.get('name', 'Unknown')
            tool_args = tool_call.get('args', {})
            emoji = get_tool_emoji(tool_name)
            
            with st.expander(f"{emoji} {tool_name.replace('_', ' ').title()}", expanded=True):
                st.json(tool_args)
        
        # Approval buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Approve", use_container_width=True, type="primary"):
                # Approve and continue
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                current_state = hitl_graph.get_state(config)
                current_state.values['approved_tools'] = st.session_state.pending_approval
                current_state.values['human_decision'] = 'approved'
                
                # Continue execution
                events = list(hitl_graph.stream(None, config=config))
                
                # Clear pending approval
                st.session_state.pending_approval = None
                
                st.success("Tools executed successfully!")
                st.rerun()
        
        with col2:
            if st.button("Reject", use_container_width=True):
                # Reject execution
                st.session_state.pending_approval = None
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': 'Tool execution was rejected by the user.'
                })
                st.error("Tool execution rejected")
                st.rerun()
        
        with col3:
            if st.button("Provide Feedback", use_container_width=True):
                st.session_state.show_feedback_modal = True

# Feedback modal
if st.session_state.get('show_feedback_modal', False):
    with st.form("feedback_form"):
        st.markdown("### Provide Feedback to AI")
        feedback = st.text_area(
            "What should the AI do instead?",
            placeholder="Example: Don't delete the file, just archive it instead"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Submit Feedback", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)
        
        if submit and feedback:
            st.session_state.pending_approval = None
            st.session_state.show_feedback_modal = False
            
            # Add feedback to messages
            st.session_state.messages.append({
                'role': 'assistant',
                'content': 'Previous action rejected. Processing your feedback...'
            })
            st.session_state.messages.append({
                'role': 'user',
                'content': f"Feedback: {feedback}"
            })
            
            st.rerun()
        
        if cancel:
            st.session_state.show_feedback_modal = False
            st.rerun()

# **************************************** Chat Interface *************************

# Display chat history
st.markdown("### Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Process with AI
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            # Stream events
            events = list(hitl_graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            ))
            
            # Get final state
            final_state = hitl_graph.get_state(config)
            
            # Check if we need approval
            if final_state.values.get('pending_tool_calls'):
                pending = final_state.values.get('pending_tool_calls', [])
                dangerous_pending = [t for t in pending if t.get('name') in DANGEROUS_TOOLS]
                
                if dangerous_pending:
                    st.session_state.pending_approval = dangerous_pending
                    response_text = "**Waiting for your approval to execute tools...**"
                else:
                    # Auto-approved safe tools
                    response_text = "Executing approved tools..."
            else:
                # Get last AI message
                last_msg = final_state.values.get('messages', [])[-1]
                response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            
            st.markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
    
    st.rerun()

# **************************************** Quick Actions *************************

st.markdown("---")
st.markdown("### Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Send Email", use_container_width=True):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'Send an email to team@company.com about the project update'
        })
        st.rerun()

with col2:
    if st.button("Create Task", use_container_width=True):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'Create a high priority task to review the code'
        })
        st.rerun()

with col3:
    if st.button("Book Meeting", use_container_width=True):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'Book a meeting with the engineering team for tomorrow'
        })
        st.rerun()

with col4:
    if st.button("Delete File", use_container_width=True):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'Delete the file temp_data.csv'
        })
        st.rerun()

# **************************************** Footer *************************

st.markdown("---")
st.caption("ip: Dangerous operations (email, delete) will require your approval before execution")