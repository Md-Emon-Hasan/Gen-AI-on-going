from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import sqlite3
import os
import asyncio

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
)

# **************************************** RAG Setup *************************

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector store path
VECTOR_DB_PATH = "./chroma_db"

# Global vector store
vector_store = None

def initialize_vector_store():
    """Initialize or load existing vector store"""
    global vector_store
    
    try:
        if os.path.exists(VECTOR_DB_PATH):
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            print(f"Loaded existing vector store with {vector_store._collection.count()} documents")
        else:
            # Create new vector store
            vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            print("Created new vector store")
    except Exception as e:
        print(f"Vector store initialization error: {e}")

# Initialize vector store on startup
initialize_vector_store()

def add_documents_to_vectorstore(file_paths: list):
    """Add documents to vector store"""
    global vector_store
    
    documents = []
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if documents:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        vector_store.add_documents(splits)
        vector_store.persist()
        
        print(f"Added {len(splits)} chunks to vector store")
        return len(splits)
    
    return 0

def load_directory_to_vectorstore(directory_path: str):
    """Load all documents from a directory"""
    global vector_store
    
    try:
        # Load PDFs
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        # Load text files
        txt_loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        pdf_docs = pdf_loader.load()
        txt_docs = txt_loader.load()
        
        all_docs = pdf_docs + txt_docs
        
        if all_docs:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)
            
            # Add to vector store
            vector_store.add_documents(splits)
            vector_store.persist()
            
            print(f"Loaded directory: {len(splits)} chunks added")
            return len(splits)
        
    except Exception as e:
        print(f"Error loading directory: {e}")
    
    return 0

# **************************************** RAG Tool *************************

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information using RAG.
    Use this when the user asks about documents or information that might be in the knowledge base.
    
    Args:
        query: The search query
    """
    global vector_store
    
    if not vector_store or vector_store._collection.count() == 0:
        return "Knowledge base is empty. Please upload documents first."
    
    try:
        # Search vector store
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Format results
        context = "Found relevant information:\n\n"
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            context += f"Source {i}: {source}\n"
            context += f"{doc.page_content}\n\n"
        
        return context
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

@tool
def add_document_to_kb(file_path: str) -> str:
    """
    Add a document (PDF or TXT) to the knowledge base.
    
    Args:
        file_path: Path to the document file
    """
    try:
        chunks = add_documents_to_vectorstore([file_path])
        return f"Successfully added document to knowledge base. {chunks} chunks indexed."
    except Exception as e:
        return f"Error adding document: {str(e)}"

@tool
def get_kb_stats() -> str:
    """Get statistics about the knowledge base."""
    global vector_store
    
    if not vector_store:
        return "Knowledge base not initialized."
    
    try:
        count = vector_store._collection.count()
        return f"Knowledge base contains {count} document chunks."
    except Exception as e:
        return f"Error getting stats: {str(e)}"

# **************************************** Other Tools *************************

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
        items = os.listdir(dirpath)
        files = [f for f in items if os.path.isfile(os.path.join(dirpath, f))]
        dirs = [d for d in items if os.path.isdir(os.path.join(dirpath, d))]
        
        result = f"Directory: {dirpath}\n\n"
        result += f"Directories ({len(dirs)}):\n"
        result += "\n".join([f"  - {d}/" for d in dirs[:10]])
        result += f"\n\nFiles ({len(files)}):\n"
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
tools = [
    search_knowledge_base,
    add_document_to_kb, 
    get_kb_stats,
    tavily_tool,
    wikipedia,
    read_file,
    list_directory,
    get_current_time
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# **************************************** Graph Setup *************************

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

def get_kb_document_count():
    """Get number of documents in knowledge base"""
    global vector_store
    if vector_store:
        try:
            return vector_store._collection.count()
        except:
            return 0
    return 0

# Export functions for frontend
__all__ = [
    'chatbot',
    'retrieve_all_threads',
    'get_available_tools',
    'add_documents_to_vectorstore',
    'load_directory_to_vectorstore',
    'get_kb_document_count',
    'vector_store'
]