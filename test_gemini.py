
import os
from google import genai
from dotenv import load_dotenv

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Test alternative models
models = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash"
]

for model in models:

    print("\nTesting model:", model)

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply only: Gemini API connection successful."
        )

        print("SUCCESS!")
        print("Model:", model)
        print("Response:", response.text)

        break

    except Exception as e:

        print("FAILED!")
        print("Error:", str(e))

else:
    print("\nBoth models failed.")
    print("Please check the API service availability.")
