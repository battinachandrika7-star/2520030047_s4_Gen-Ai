
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Step 1: Load API key from .env
load_dotenv()

# Step 2: Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )

# Step 3: Initialize Gemini client
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
)

print("Connecting to Gemini...", flush=True)

try:
    # Step 4: Send prompt to Gemini
    response = client.models.generate_content(
        model="gemini-3.8-flash",
       contents="Explain Generative AI with two real-world examples."
    )

    # Step 5: Display response
    print("\nResponse received!", flush=True)
    print("\nLLM Response:")
    print(response.text)

except Exception as error:
    print("\nError occurred:")
    print(error)
