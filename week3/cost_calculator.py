
# cost_calculator.py

# Example rates per 1 million tokens.
# These are sample values from the Week 3 document.
# Replace them with verified rates when needed.

PRICING = {
    "gemini-3.8-flash": {
        "input": 0.50,
        "output": 2.00
    }
}

def calculate_cost(provider, model, input_tokens, output_tokens):

    # Ollama runs locally and has no per-token API fee.
    if provider == "OllamaProvider":
        return 0.0

    rates = PRICING.get(model)

    if rates is None:
        return None

    input_cost = (
        input_tokens * rates["input"]
    ) / 1_000_000

    output_cost = (
        output_tokens * rates["output"]
    ) / 1_000_000

    total_cost = input_cost + output_cost

    return round(total_cost, 10)
