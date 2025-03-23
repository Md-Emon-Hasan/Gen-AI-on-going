from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

# Load environment variables
load_dotenv()

# Define prompt templates
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define output parser
parser = StrOutputParser()

# Create report generation chain
report_gen_chain = prompt1 | model | parser

# Create branch chain based on text length
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

# Final chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Test the chain with sample input
print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))
