import os
from langchain_community.document_loaders import PyPDFLoader

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to your PDF file
pdf_path = os.path.join(script_dir, 'dl-curriculum.pdf')

# Initialize the loader with the relative path
loader = PyPDFLoader(pdf_path)

# Load the document
docs = loader.load()

# Print the number of documents loaded
print(len(docs))

# Print the content of the first page
print(docs[0].page_content)

# Print the metadata of the first document
print(docs[0].metadata)