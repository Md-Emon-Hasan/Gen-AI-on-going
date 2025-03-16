from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Hugging Face embeddings (no API key needed)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Australian cricket content
documents = [
    "Ricky Ponting is one of Australia's greatest captains and batsmen, leading the team to multiple World Cup victories.",
    "Steve Smith is an Australian cricketer known for his unorthodox batting technique and remarkable consistency.",
    "David Warner is a destructive Australian opener, famous for his aggressive batting style.",
    "Pat Cummins is Australia's fast bowler and captain, known for his lethal pace and leadership skills.",
    "Glenn Maxwell is an Australian all-rounder, known for his explosive batting in limited-overs cricket."
]

# Query related to Australian cricket
query = 'Tell me about David Warner'

# Generate embeddings for documents and query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the most relevant document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Output the results
print("Query:", query)
print("Most relevant document:", documents[index])
print("Similarity score:", score)
