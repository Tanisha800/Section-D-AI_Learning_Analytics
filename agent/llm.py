print("LLM FILE LOADED")
import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


client = None
if api_key:
    try:
        client = Groq()
        print("Groq client initialized")
    except Exception as e:
        print("Failed to initialize Groq client:", e)
else:
    print("GROQ_API_KEY not found")

print("API KEY VALUE:", api_key)
def generate_response(prompt):
    """
    Generates response from Groq LLM.
    Handles all errors gracefully to prevent app crashes.
    """


    if client is None:
        return "GROQ API key not set. Please configure it in Render Environment Variables."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast + good
            messages=[
                {"role": "system", "content": "You are an AI Study Coach that helps students improve learning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000,
        )

        return response.choices[0].message.content

    except Exception as e:
        print("LLM ERROR:", e)
        return f"LLM Error: {str(e)}"



