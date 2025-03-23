from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

# Load environment variables
load_dotenv()

# Define prompt templates
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define output parser
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Create the processing chain
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# Test the chain with sample input
print(chain.invoke({'topic': 'AI'}))
