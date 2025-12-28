import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads, get_available_tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# **************************************** Utility Functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    messages = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
    
    # Convert to display format
    temp_messages = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, HumanMessage):
            temp_messages.append({'role': 'user', 'content': msg.content})
            i += 1
        elif isinstance(msg, AIMessage):
            # Check if this AI message used tools
            tools_used = []
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'Unknown')
                    emoji = get_tool_emoji(tool_name)
                    formatted_name = format_tool_name(tool_name)
                    tools_used.append(f'{emoji} {formatted_name}')
            
            # Get the content
            content = msg.content if msg.content else ""
            
            # Add tool references if any
            if tools_used:
                content += f"\n\n---\n** Tools used:** {', '.join(tools_used)}"
            
            temp_messages.append({'role': 'assistant', 'content': content})
            i += 1
        else:
            # Skip tool messages
            i += 1
    
    return temp_messages

def get_tool_emoji(tool_name):
    """Get emoji for tool type"""
    tool_emojis = {
        'tavily_search_results_json': '',
        'wikipedia': '',
        'read_file': '',
        'list_directory': '',
        'get_current_time': '',
    }
    return tool_emojis.get(tool_name, '')

def format_tool_name(tool_name):
    """Format tool name for display"""
    name_map = {
        'tavily_search_results_json': 'Tavily Web Search',
        'wikipedia': 'Wikipedia',
        'read_file': 'File Reader',
        'list_directory': 'Directory Listing',
        'get_current_time': 'Current Time',
    }
    return name_map.get(tool_name, tool_name.replace('_', ' ').title())

# **************************************** Session Setup ******************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'show_tools' not in st.session_state:
    st.session_state['show_tools'] = False

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')
st.sidebar.markdown('*with Web Search, Wikipedia & MCP Tools*')

if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

# Tool Information
with st.sidebar.expander("Available Tools"):
    tools_list = get_available_tools()
    for tool in tools_list:
        emoji = get_tool_emoji(tool['name'])
        st.markdown(f"**{emoji} {format_tool_name(tool['name'])}**")
        st.caption(tool['description'][:100] + "...")

st.sidebar.markdown("---")
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(f"{str(thread_id)[:8]}...", key=thread_id, use_container_width=True):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)
        st.rerun()

# **************************************** Main UI ************************************

st.title('AI Chatbot with MCP Tools')
st.markdown('Ask me anything! I can search the web, Wikipedia, read files, and more.')

# Quick actions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Search Web", use_container_width=True):
        st.session_state['quick_action'] = "Search the web for latest news about AI"
with col2:
    if st.button("Search Wiki", use_container_width=True):
        st.session_state['quick_action'] = "Search Wikipedia for information about Python programming"
with col3:
    if st.button("List Files", use_container_width=True):
        st.session_state['quick_action'] = "List files in the current directory"

# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle quick actions
if 'quick_action' in st.session_state:
    user_input = st.session_state['quick_action']
    del st.session_state['quick_action']
    
    # Display user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    # Display assistant response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        tool_status = st.empty()
        full_response = ""
        tools_used_set = set()
        
        for chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ):
            if isinstance(chunk, AIMessage):
                # Collect tool calls
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        tool_name = tc.get('name', 'Unknown')
                        emoji = get_tool_emoji(tool_name)
                        formatted_name = format_tool_name(tool_name)
                        tools_used_set.add(f'{emoji} {formatted_name}')
                        tool_status.info(f"Using: {', '.join(tools_used_set)}")
                
                # Collect content
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
        
        tool_status.empty()
        
        # Add tool references at the end
        if tools_used_set:
            full_response += f"\n\n---\n** Tools used:** {', '.join(sorted(tools_used_set))}"
        
        response_placeholder.markdown(full_response)
    
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
    st.rerun()

# User input
user_input = st.chat_input('Type your message here...')

if user_input:
    # Display user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Display assistant response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        tool_status = st.empty()
        full_response = ""
        tools_used_set = set()
        
        # Stream the response
        for chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ):
            if isinstance(chunk, AIMessage):
                # Collect tool calls
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        tool_name = tc.get('name', 'Unknown')
                        emoji = get_tool_emoji(tool_name)
                        formatted_name = format_tool_name(tool_name)
                        tools_used_set.add(f'{emoji} {formatted_name}')
                        tool_status.info(f" Using: {', '.join(tools_used_set)}")
                
                # Collect content
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
        
        tool_status.empty()
        
        # Add tool references at the end
        if tools_used_set:
            full_response += f"\n\n---\n**Tools used:** {', '.join(sorted(tools_used_set))}"
        
        response_placeholder.markdown(full_response)

    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})