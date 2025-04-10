import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google Embeddings wrapper

# Load the .env file to get your API key
load_dotenv()

# Initialize Google Embeddings using API key from environment variables
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the SemanticChunker with the Google embeddings
text_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

# Sample input text
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Split the text semantically
docs = text_splitter.create_documents([sample])

# Output the number of documents and their content
print(f"Total chunks: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc.page_content}\n")
