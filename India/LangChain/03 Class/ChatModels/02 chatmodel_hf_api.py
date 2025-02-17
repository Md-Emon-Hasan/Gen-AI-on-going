from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set your Hugging Face API key
HUGGINGFACEHUB_API_TOKEN = "your_api_key_here"  # Replace with your actual API key

# Initialize the Hugging Face model with the API key
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Use the model for chat
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke("What is the capital of India")

# Print the result
print(result.content)