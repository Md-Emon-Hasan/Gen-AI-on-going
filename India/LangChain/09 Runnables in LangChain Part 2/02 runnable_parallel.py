from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

# Load environment variables
load_dotenv()

# Define prompt templates
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define output parser
parser = StrOutputParser()

# Create parallel processing chain
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

# Test the chain with sample input
result = parallel_chain.invoke({'topic': 'AI'})

print(result['tweet'])
print(result['linkedin'])