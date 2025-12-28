import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
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
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

def display_message(message):
    """Display message with appropriate formatting"""
    if isinstance(message, HumanMessage):
        role = 'user'
        content = message.content
    elif isinstance(message, ToolMessage):
        # Skip tool messages in display
        return None
    else:
        role = 'assistant'
        content = message.content
        
        # Check if message has tool calls (references)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_info = []
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', 'Unknown Tool')
                tool_info.append(f"Using: {tool_name}")
            
            if tool_info:
                content = content + "\n\n" + "\n".join(tool_info)
    
    return {'role': role, 'content': content}

# **************************************** Session Setup ******************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')
st.sidebar.markdown('*with Tavily & Wikipedia Search*')

if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(f"{str(thread_id)[:8]}...", key=thread_id, use_container_width=True):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            displayed_msg = display_message(msg)
            if displayed_msg:
                temp_messages.append(displayed_msg)

        st.session_state['message_history'] = temp_messages
        st.rerun()

# **************************************** Main UI ************************************

st.title('AI Chatbot with Search')
st.markdown('Ask me anything! I can search the web and Wikipedia for you.')

# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

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
        full_response = ""
        
        # Stream the response
        for chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ):
            # Only process AI messages (skip tool messages)
            if isinstance(chunk, AIMessage) and chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Get the complete state to check for tool usage
        final_state = chatbot.get_state(config=CONFIG)
        last_message = final_state.values['messages'][-1]
        
        # Add reference information if tools were used
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            references = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get('name', 'Unknown')
                if tool_name == 'tavily_search_results_json':
                    references.append('Tavily Web Search')
                elif tool_name == 'wikipedia':
                    references.append('Wikipedia')
            
            if references:
                full_response += "\n\n---\n**Sources used:** " + ", ".join(references)
                response_placeholder.markdown(full_response)

    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})