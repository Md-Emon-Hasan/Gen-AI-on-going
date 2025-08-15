import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph_backend import chatbot

# ================= Utilities =================
def generate_thread_id():
    return uuid.uuid4()

def add_thread(thread_id):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    state_ctx = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state_ctx.values.get("messages", [])

def lc_to_ui(messages: list[BaseMessage]):
    out = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = getattr(msg, "type", "assistant")
        out.append({"role": role, "content": msg.content})
    return out

# ================= Session Init =================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])

# ================= Sidebar =================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        msgs = load_conversation(thread_id)
        st.session_state["message_history"] = lc_to_ui(msgs)

# ================= Main UI =================
st.caption(f"Active thread: `{st.session_state['thread_id']}`")

# Show history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type here")

if user_input:
    # Show user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": str(st.session_state["thread_id"])}}

    # Call chatbot (non-streaming)
    result_state = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
    )

    # Extract last AI message safely
    messages = result_state.get("messages", [])
    ai_text = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            ai_text = msg.content
            break

    if ai_text is None:
        ai_text = "_No AI response found._"

    # Show AI reply
    with st.chat_message("assistant"):
        st.markdown(ai_text)

    # Save to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_text})


    
# streamlit run "C:/Users/emon1/Desktop/Gen AI on going/India/Langgraph/12 build a Resume Chat feature like ChatGPT/streamlit_frontend_threading.py"