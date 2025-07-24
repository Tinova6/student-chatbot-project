from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to fetch from it

# --- Configure your Gemini API Key ---
# RECOMMENDED: Set your API key as an environment variable
# For example, in your terminal before running:
# export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
#
# OR, for quick testing (less secure for production):
# Replace "YOUR_GEMINI_API_KEY" with your actual API key obtained from Google Cloud Console
# You can get one here: https://console.cloud.google.com/ai/generativelanguage/api
# Make sure the Generative Language API is enabled for your project.
GEMINI_API_KEY = os.getenv("AIzaSyACK-u37YVz9HmjWTC7LrPGr8cb5DnRwkQ", "AIzaSyACK-u37YVz9HmjWTC7LrPGr8cb5DnRwkQ")

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set or is using the placeholder. "
          "Please set it as an environment variable (GOOGLE_API_KEY) "
          "or replace 'YOUR_GEMINI_API_KEY' in app.py with your actual key.")
    # You might want to exit or raise an error in a production environment
    # sys.exit("API Key not configured.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Generative Model
# Using 'gemini-pro' for text-only generation.
# If you need image input, you'd use 'gemini-pro-vision'.
model = genai.GenerativeModel('gemini-pro')

# Serve the frontend HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Generate content using the Gemini model
        # For a simple request-response, we just send the user's message.
        # For a conversational history, you would maintain and pass a list of messages.
        response = model.generate_content(user_message)

        # Access the text from the response
        # The exact structure might vary, so add error handling
        ai_response_text = "Sorry, I couldn't generate a response."
        if response and response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            ai_response_text = part.text
                            break # Take the first text part
                if ai_response_text != "Sorry, I couldn't generate a response.":
                    break

        return jsonify({"response": ai_response_text})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for automatic reloading on code changes
    # host='0.0.0.0' makes the server accessible from other devices on your network
    app.run(debug=True, host='127.0.0.1', port=5000)
