import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Gemini client with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-pro')

# Define a function to get a response from Gemini 1.5 Pro
def get_response_from_gemini(question, text):
    prompt = f"Answer the following question: \n{question}\n from the following text: \n{text}"
    
    # Generate content using the model
    response = model.generate_content(prompt)
    
    # Return the text part of the response
    return response.text

# Define the prompt template (though we're not using it in this version)
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

# URL of the page to load
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

# Load documents from the URL
docs = loader.load()

# Use the function to get a response from Gemini
try:
    response_text = get_response_from_gemini(
        'What is the product that we are talking about?', 
        docs[0].page_content
    )
    print(response_text)
except Exception as e:
    print(f"An error occurred: {str(e)}")