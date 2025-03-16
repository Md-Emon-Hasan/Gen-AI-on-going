from langchain_google_genai import ChatGoogleGenerativeAI

# Set the API key directly
GOOGLE_API_KEY = "your_api_key_here"  # Replace with your actual API key

# Initialize the model with the API key
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', google_api_key=GOOGLE_API_KEY)

# Invoke the model
result = model.invoke('What is the capital of India')

# Print the result
print(result.content)