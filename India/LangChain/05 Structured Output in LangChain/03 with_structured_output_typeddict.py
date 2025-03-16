import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the structured schema
class Review(TypedDict):
    key_themes: list[str]
    summary: str
    sentiment: Literal["pos", "neg", "neutral"]
    pros: Optional[list[str]]
    cons: Optional[list[str]]
    name: Optional[str]

# Initialize Gemini Pro
model = genai.GenerativeModel("gemini-1.5-pro")

# Review text to analyze
review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor (great for gaming and productivity)
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

# Define the prompt for structured output
prompt = f"""
Analyze the following product review and extract structured data:

Review:
{review_text}

Return the information in JSON format with the following structure:
- key_themes: A list of key topics discussed in the review.
- summary: A brief 2-3 sentence summary.
- sentiment: Either "pos", "neg", or "neutral".
- pros: A list of advantages mentioned.
- cons: A list of disadvantages mentioned.
- name: The name of the reviewer if available.
"""

# Invoke Gemini Pro
response = model.generate_content(prompt)
parsed_result = response.text  # Gemini returns plain text, so we need to parse it.

# Print the extracted information
print("Extracted Review Data:\n", parsed_result)
