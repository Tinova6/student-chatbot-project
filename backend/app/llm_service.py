import google.generativeai as genai
from .config import GEMINI_API_KEY

# Configure the Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-pro')

async def get_gemini_response(prompt: str):
    """
    Sends a prompt to the Gemini model and returns the text response.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return "I'm sorry, I'm having trouble connecting right now. Please try again later."