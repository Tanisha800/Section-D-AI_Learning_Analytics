from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an AI Study Coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=30000
    )

    return response.choices[0].message.content