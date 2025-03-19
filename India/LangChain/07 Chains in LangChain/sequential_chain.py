from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Define the first prompt template
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Define the second prompt template
prompt2 = PromptTemplate(
    template='Generate a 5-pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define the output parser
parser = StrOutputParser()

# Create the processing chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke the chain with a topic
result = chain.invoke({'topic': 'Unemployment in India'})

# Print the result
print(result)

# Print the chain graph
chain.get_graph().print_ascii()