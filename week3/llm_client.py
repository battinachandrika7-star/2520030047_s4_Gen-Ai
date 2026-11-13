
# llm_client.py

from abc import ABC, abstractmethod
from google import genai
from google.genai import types
from google.genai import errors
import requests
import time

from cost_calculator import calculate_cost


# =====================================
# 1. LLM PROVIDER INTERFACE
# =====================================

class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# =====================================
# 2. GEMINI CLOUD AI PROVIDER
# =====================================

class GeminiProvider(LLMProvider):

    def __init__(self, api_key):

        if not api_key:
            raise ValueError("Gemini API key is missing.")

        self.model = "gemini-3.8-flash"

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=120000
            )
        )

    def generate(self, prompt):

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=256
                )
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            usage = response.usage_metadata

            input_tokens = (
                usage.prompt_token_count if usage else 0
            )

            output_tokens = (
                usage.candidates_token_count if usage else 0
            )

            return {
                "response": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

        except errors.APIError as error:
            print(
                f"Gemini API error ({error.code}): {error.message}"
            )
            raise


# =====================================
# 3. OLLAMA LOCAL AI PROVIDER
# =====================================

class OllamaProvider(LLMProvider):

    def __init__(self, model="qwen2.5:1.5b"):

        self.model = model

        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return {
            "response": data["response"],
            "input_tokens": data.get(
                "prompt_eval_count", 0
            ),
            "output_tokens": data.get(
                "eval_count", 0
            )
        }


# =====================================
# 4. COMMON LLM CLIENT
# =====================================

class LLMClient:

    def __init__(self, provider, logger):

        self.provider = provider
        self.logger = logger

    def generate(self, prompt):

        start_time = time.perf_counter()

        try:

            # Generate AI response
            result = self.provider.generate(prompt)

            latency = round(
                (time.perf_counter() - start_time) * 1000,
                2
            )

            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]

            provider_name = type(self.provider).__name__
            model = self.provider.model

            # Calculate API cost
            cost = calculate_cost(
                provider_name,
                model,
                input_tokens,
                output_tokens
            )

            # Prepare monitoring information
            log_data = {
                "provider": provider_name,
                "model": model,
                "prompt": prompt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency,
                "cost": cost,
                "attempts": 1,
                "status": "success",
                "error": None
            }

            try:
                self.logger.log_call(log_data)
                print("Call details saved in MongoDB!")

            except Exception as log_error:
                print("MongoDB logging error:", log_error)

            return result["response"]

        except Exception as error:

            latency = round(
                (time.perf_counter() - start_time) * 1000,
                2
            )

            failed_log = {
                "provider": type(self.provider).__name__,
                "model": self.provider.model,
                "prompt": prompt,
                "latency_ms": latency,
                "attempts": 1,
                "status": "failed",
                "error": str(error)
            }

            try:
                self.logger.log_call(failed_log)
                print("Failed call details saved in MongoDB!")

            except Exception as log_error:
                print("MongoDB logging error:", log_error)

            raise
