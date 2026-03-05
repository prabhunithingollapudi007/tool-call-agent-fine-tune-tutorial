"""
tools.py — Dummy tool/function definitions that simulate real API calls.

These functions are the "tools" our fine-tuned model will learn to call.
Each function returns mock data — in a real application, these would call
actual APIs (weather services, databases, search engines, etc.).

HOW IT WORKS:
- Each function has a clear signature with type hints
- TOOL_REGISTRY maps function names to their callables
- TOOL_DEFINITIONS provides the JSON schema the model sees during training/inference
"""

import json
import random
from datetime import datetime


# ============================================================
# Tool Functions (mock implementations)
# ============================================================

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather information for a given city."""
    # Mock weather data — in production, this would call a weather API
    weather_data = {
        "New York": {"temp_c": 18, "condition": "partly cloudy", "humidity": 65},
        "London": {"temp_c": 14, "condition": "rainy", "humidity": 80},
        "Tokyo": {"temp_c": 26, "condition": "sunny", "humidity": 55},
        "Paris": {"temp_c": 20, "condition": "overcast", "humidity": 70},
        "Sydney": {"temp_c": 22, "condition": "sunny", "humidity": 50},
    }

    default = {"temp_c": random.randint(10, 35), "condition": "clear", "humidity": random.randint(30, 90)}
    data = weather_data.get(city, default)

    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity": data["humidity"],
    }


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression and return the result."""
    # Safe evaluation of math expressions
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return {"expression": expression, "result": None, "error": "Invalid characters in expression"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


def search_web(query: str, num_results: int = 3) -> dict:
    """Search the web for information (mock results)."""
    # Mock search results
    mock_results = [
        {
            "title": f"Result about: {query}",
            "snippet": f"This is a comprehensive overview of {query}. "
                       f"According to recent studies, {query} has been a topic of significant interest...",
            "url": f"https://example.com/article/{query.replace(' ', '-').lower()}",
        },
        {
            "title": f"{query} - Wikipedia",
            "snippet": f"{query} refers to a concept widely discussed in various fields. "
                       f"It encompasses several key aspects that researchers have explored...",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
        },
        {
            "title": f"Understanding {query} | Expert Guide",
            "snippet": f"A detailed guide to understanding {query}. "
                       f"This resource covers fundamentals, applications, and recent developments...",
            "url": f"https://example.com/guide/{query.replace(' ', '-').lower()}",
        },
    ]
    return {"query": query, "results": mock_results[:num_results]}


def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a given ticker symbol."""
    # Mock stock data
    stocks = {
        "AAPL": {"price": 178.50, "change": 2.30, "change_pct": 1.31},
        "GOOGL": {"price": 141.80, "change": -0.95, "change_pct": -0.67},
        "MSFT": {"price": 378.20, "change": 4.10, "change_pct": 1.10},
        "TSLA": {"price": 245.60, "change": -3.40, "change_pct": -1.37},
        "AMZN": {"price": 185.30, "change": 1.20, "change_pct": 0.65},
    }

    symbol = symbol.upper()
    if symbol in stocks:
        data = stocks[symbol]
    else:
        data = {
            "price": round(random.uniform(10, 500), 2),
            "change": round(random.uniform(-5, 5), 2),
            "change_pct": round(random.uniform(-3, 3), 2),
        }

    return {"symbol": symbol, "currency": "USD", **data}


def get_current_datetime(timezone: str = "UTC") -> dict:
    """Get the current date and time."""
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


def translate_text(text: str, target_language: str, source_language: str = "auto") -> dict:
    """Translate text from one language to another (mock)."""
    # Mock translation — just returns a placeholder
    translations = {
        "spanish": "Hola, este es un texto traducido de ejemplo.",
        "french": "Bonjour, ceci est un exemple de texte traduit.",
        "german": "Hallo, dies ist ein Beispiel für übersetzten Text.",
        "japanese": "こんにちは、これは翻訳されたサンプルテキストです。",
        "chinese": "你好，这是一个翻译示例文本。",
    }

    translated = translations.get(
        target_language.lower(),
        f"[Translated to {target_language}]: {text}"
    )

    return {
        "original_text": text,
        "translated_text": translated,
        "source_language": source_language,
        "target_language": target_language,
    }


# ============================================================
# Tool Registry — maps function names to callables
# ============================================================

TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web,
    "get_stock_price": get_stock_price,
    "get_current_datetime": get_current_datetime,
    "translate_text": translate_text,
}


# ============================================================
# Tool Definitions — JSON schema the model sees
# ============================================================

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a given city. Returns temperature, condition, and humidity.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name to get weather for"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit", "default": "celsius"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result. Supports basic arithmetic: +, -, *, /, parentheses.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g. '(2 + 3) * 4'"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information on a given query. Returns a list of relevant results with titles, snippets, and URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {"type": "integer", "description": "Number of results to return (1-5)", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol. Returns price, change, and percentage change.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time, including day of the week.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone name (e.g., 'UTC', 'US/Eastern')", "default": "UTC"},
            },
            "required": [],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text from one language to another. Supports multiple target languages.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate"},
                "target_language": {"type": "string", "description": "Target language (e.g., 'spanish', 'french', 'german', 'japanese', 'chinese')"},
                "source_language": {"type": "string", "description": "Source language (default: auto-detect)", "default": "auto"},
            },
            "required": ["text", "target_language"],
        },
    },
]


def get_tool_definitions_json() -> str:
    """Return tool definitions as a formatted JSON string for the system prompt."""
    return "\n".join(json.dumps(tool, indent=2) for tool in TOOL_DEFINITIONS)


def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a tool by name with the given arguments. Returns the result dict."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**arguments)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    print("=== Testing all tools ===\n")

    print("1. get_weather('Tokyo'):")
    print(f"   {get_weather('Tokyo')}\n")

    print("2. calculate('(2 + 3) * 4'):")
    print(f"   {calculate('(2 + 3) * 4')}\n")

    print("3. search_web('machine learning', 2):")
    result = search_web("machine learning", 2)
    print(f"   Found {len(result['results'])} results\n")

    print("4. get_stock_price('AAPL'):")
    print(f"   {get_stock_price('AAPL')}\n")

    print("5. get_current_datetime():")
    print(f"   {get_current_datetime()}\n")

    print("6. translate_text('Hello', 'spanish'):")
    print(f"   {translate_text('Hello', 'spanish')}\n")

    print("=== All tools working! ===")
