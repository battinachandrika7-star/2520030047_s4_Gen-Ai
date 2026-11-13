
# main.py

import os

from dotenv import load_dotenv

from llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMClient
)

from mongo_logger import MongoLogger


# Load environment variables
load_dotenv(override=True)


print("================================")
print("     WEEK 3: LLM MONITORING")
print("================================")

print("1. Gemini")
print("2. Ollama")

choice = input("\nSelect provider (1/2): ").strip()

logger = None

try:

    # Connect to MongoDB
    logger = MongoLogger()

    # Select AI provider
    if choice == "1":

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key not found in .env"
            )

        provider = GeminiProvider(api_key)

    elif choice == "2":

        provider = OllamaProvider("qwen2.5:1.5b")

    else:

        raise ValueError(
            "Invalid selection. Choose 1 or 2."
        )

    # Create common client
    client = LLMClient(provider, logger)

    # Get user prompt
    prompt = input("\nEnter your question: ")

    # Generate response
    answer = client.generate(prompt)

    print("\n========== AI RESPONSE ==========")
    print(answer)

    print("\n========== COMPLETED ==========")

except Exception as error:

    print("\nApplication error:", error)

finally:

    if logger is not None:
        logger.close()
