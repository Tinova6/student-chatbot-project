import os
from flask import Flask, request, jsonify
from flask_cors import CORS
# For real LLM integration:
# import google.generativeai as genai
# from openai import OpenAI

chatbot = Flask(__name__)
CORS(chatbot) # Enable CORS for frontend communication

# --- LLM Configuration (Uncomment and configure for actual LLM) ---
# For Google Gemini:
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# LLM_MODEL = genai.GenerativeModel('gemini-pro')

# For OpenAI:
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# LLM_MODEL_NAME = "gpt-3.5-turbo"

# --- Define our "Specialized Agents" (as placeholder functions) ---
def wellness_agent(query):
    """Simulates a wellness coach's response."""
    # In a real app, this would involve RAG on wellness data,
    # or calling specific wellness tools/APIs.
    return f"As your Wellness Coach, I understand you're interested in: '{query}'. Let's talk about self-care and mental well-being!"

def learning_agent(query):
    """Simulates a learning assistant's response."""
    # In a real app, this would involve RAG on educational content,
    # or fetching learning resources.
    return f"Welcome to the Learning Zone! Your query about '{query}' suggests a deep dive into knowledge. What topic are you exploring today?"

def creative_muse_agent(query):
    """Simulates a creative muse's response."""
    # In a real app, this might involve generating poetry, stories, or image prompts.
    return f"Ah, a fellow creative! '{query}' inspires me. Let's craft something beautiful or imagine new worlds together."

def professional_agent(query):
    """Simulates a professional assistant's response."""
    # In a real app, this would integrate with calendars, task managers,
    # or professional knowledge bases.
    return f"In the professional realm, you're asking about '{query}'. How can I help you be more productive or effective?"

def general_agent(query):
    """Handles queries that don't fit into specific categories."""
    return f"I'm a general assistant. You asked: '{query}'. How else can I assist you with everyday tasks or information?"

# --- LLM-based Router Logic ---
def route_query_with_llm(query):
    """
    Uses an LLM to determine the user's intent and route the query.
    In a real scenario, the LLM would be prompted to output a specific
    tag or JSON that maps to one of our agent functions.
    """
    # Define the possible "destinations" for the LLM to choose from
    # This prompt is crucial for LLM-based routing
    routing_prompt = f"""
    Analyze the following user query and determine which of the following categories it best fits.
    Respond ONLY with the category name. If none fit perfectly, choose 'general'.

    Categories:
    - wellness (e.g., stress, diet, exercise, mental health, self-care)
    - learning (e.g., explain, teach, understand, concept, study, learn about)
    - creative (e.g., write a poem, story, imagine, brainstorm, art, design, generate)
    - professional (e.g., schedule, meeting, productivity, career, task, work)
    - general (for anything else or casual chat)

    User Query: "{query}"

    Category:
    """

    try:
        # --- Simulate LLM response for routing ---
        # In a real implementation, you'd call your LLM here:
        # For Gemini:
        # response = LLM_MODEL.generate_content(routing_prompt).text.strip().lower()
        # For OpenAI:
        # response = client.chat.completions.create(
        #     model=LLM_MODEL_NAME,
        #     messages=[{"role": "user", "content": routing_prompt}]
        # ).choices[0].message.content.strip().lower()

        # --- Placeholder for LLM Routing (remove when integrating real LLM) ---
        print(f"\nDEBUG: Routing LLM Prompt:\n{routing_prompt}")
        # Simple keyword-based simulation for demonstration without a real LLM call
        query_lower = query.lower()
        if "stress" in query_lower or "diet" in query_lower or "exercise" in query_lower or "meditate" in query_lower:
            predicted_category = "wellness"
        elif "explain" in query_lower or "teach" in query_lower or "learn" in query_lower or "concept" in query_lower:
            predicted_category = "learning"
        elif "poem" in query_lower or "story" in query_lower or "imagine" in query_lower or "brainstorm" in query_lower or "create" in query_lower:
            predicted_category = "creative"
        elif "schedule" in query_lower or "meeting" in query_lower or "work" in query_lower or "career" in query_lower or "productivity" in query_lower:
            predicted_category = "professional"
        else:
            predicted_category = "general"
        print(f"DEBUG: Predicted category (simulated): {predicted_category}")
        # --- End Placeholder ---

        return predicted_category
    except Exception as e:
        print(f"Error in LLM routing: {e}")
        return "general" # Fallback to general if routing fails

@chatbot.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"response": "No message received."}), 400

    # Step 1: Route the query
    destination = route_query_with_llm(user_message)

    # Step 2: Call the appropriate agent
    bot_response = "I'm sorry, I couldn't find a specialized agent for that."
    if destination == "wellness":
        bot_response = wellness_agent(user_message)
    elif destination == "learning":
        bot_response = learning_agent(user_message)
    elif destination == "creative":
        bot_response = creative_muse_agent(user_message)
    elif destination == "professional":
        bot_response = professional_agent(user_message)
    else: # Fallback to general
        bot_response = general_agent(user_message)

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    chatbot.run(host="129.0.0.1", port=5000)