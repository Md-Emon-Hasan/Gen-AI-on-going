from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Model name for GPT-2 (or any other model you prefer)
model_name = "gpt2"

# Print message for loading model
print(f"Loading {model_name} model...")

# Load the GPT-2 model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define the LLM using HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id=model_name,  # You can specify the model name directly for Hugging Face
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)

# Define prompts
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5-line summary on the following text. /n {text}',
    input_variables=['text']
)

# First prompt -> Detailed Report
prompt1 = template1.invoke({'topic': 'black hole'})

# Get response from model
result = chat_model.invoke(prompt1)

# Second prompt -> Summary of the report
prompt2 = template2.invoke({'text': result.content})

# Get summarized response
result1 = chat_model.invoke(prompt2)

# Print the final output
print(result1.content)
