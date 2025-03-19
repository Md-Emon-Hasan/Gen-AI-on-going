from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Define the prompt template
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define the output parser
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# Invoke the chain with a topic
result = chain.invoke({'topic': 'cricket'})

# Print the result
print(result)

# Print the chain graph
chain.get_graph().print_ascii()