from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# --- Backend State for Interactive Dolls/Objects ---
# This would be managed more robustly in a real app, potentially in a database
current_atom_state = {
    "protons": 0,
    "neutrons": 0,
    "electrons": 0,
    "nucleus_stable": False,
    "electron_shells_stable": False,
    "target_atom": "hydrogen" # What atom are we building?
}

# Knowledge base for simple atom facts
atom_data = {
    "hydrogen": {"protons": 1, "neutrons": 0, "electrons": 1, "valence_electrons": 1},
    "helium": {"protons": 2, "neutrons": 2, "electrons": 2, "valence_electrons": 2},
    "carbon": {"protons": 6, "neutrons": 6, "electrons": 6, "valence_electrons": 4}
}

# Mock LLM API Call
def call_llm_api(prompt_text, current_doll_state, context=None):
    print(f"\n[DEBUG] LLM Input: '{prompt_text}', State: {current_doll_state}, Context: {context}")
    
    # --- LLM Logic Simulation (Highly Simplified) ---
    prompt_lower = prompt_text.lower()
    
    response_text = ""
    animation_trigger = None
    doll_state_update = {} # Changes to send to frontend for dolls

    if context == "building_atom":
        target = atom_data[current_doll_state["target_atom"]]
        
        # Check for user input related to placing particles
        if "place" in prompt_lower or "put" in prompt_lower or "drag" in prompt_lower:
            if "proton" in prompt_lower:
                current_doll_state["protons"] += 1
                response_text = "Good job! You've added a proton. Protons have a positive charge and go in the nucleus. What's next?"
                animation_trigger = "proton_add_animation"
                doll_state_update = {"particle_add": "proton"}
            elif "neutron" in prompt_lower:
                current_doll_state["neutrons"] += 1
                response_text = "A neutron! Neutrons are neutral and also go in the nucleus. Keep building!"
                animation_trigger = "neutron_add_animation"
                doll_state_update = {"particle_add": "neutron"}
            elif "electron" in prompt_lower:
                current_doll_state["electrons"] += 1
                response_text = "An electron! Remember, electrons orbit the nucleus in shells. Try to match the number of protons for a neutral atom."
                animation_trigger = "electron_add_animation"
                doll_state_update = {"particle_add": "electron"}
            else:
                response_text = "What kind of particle are you trying to place? Tell me if it's a proton, neutron, or electron!"
                animation_trigger = "chatbot_confused"

        # Check if nucleus is complete
        if current_doll_state["protons"] == target["protons"] and \
           current_doll_state["neutrons"] == target["neutrons"] and \
           not current_doll_state["nucleus_stable"]:
            current_doll_state["nucleus_stable"] = True
            response_text += "\nWow! Your nucleus looks stable! Now, think about the electrons. For a neutral atom, you need one electron for every proton."
            animation_trigger = "nucleus_stable_glow"
            
        # Check if atom is complete
        if current_doll_state["protons"] == target["protons"] and \
           current_doll_state["neutrons"] == target["neutrons"] and \
           current_doll_state["electrons"] == target["electrons"] and \
           current_doll_state["nucleus_stable"] and \
           not current_doll_state["electron_shells_stable"]:
            current_doll_state["electron_shells_stable"] = True
            response_text = f"Amazing! You've successfully built a {current_doll_state['target_atom'].capitalize()} atom! Great job balancing the particles!"
            animation_trigger = "atom_complete_sparkle"
            
        # If user asks to reset
        if "reset" in prompt_lower or "start over" in prompt_lower:
            global current_atom_state # Modify global state for simplicity in this demo
            current_atom_state = {
                "protons": 0, "neutrons": 0, "electrons": 0,
                "nucleus_stable": False, "electron_shells_stable": False,
                "target_atom": current_atom_state["target_atom"] # Keep target
            }
            response_text = f"Okay, let's reset the atom builder. Try building a {current_atom_state['target_atom'].capitalize()} atom again!"
            animation_trigger = "reset_build_animation"
            doll_state_update = {"reset_canvas": True} # Instruction to clear frontend

    elif "build atom" in prompt_lower or "atom builder" in prompt_lower:
        response_text = "Welcome to the Atom Builder! Let's start with a simple one: can you build a **Hydrogen** atom? Remember, Hydrogen has **1 proton**, **0 neutrons**, and **1 electron**."
        animation_trigger = "atom_builder_intro"
        context = "building_atom" # Set context for follow-up
        
    elif "explain atom" in prompt_lower or "what is an atom" in prompt_lower:
        response_text = "An atom is the basic building block of all matter. It's made of a central nucleus (with protons and neutrons) and electrons orbiting around it. We can explore more with our Atom Builder!"
        animation_trigger = "atom_concept_animation"
        
    else:
        response_text = "I'm your Concept Playground! Try asking me to 'build an atom' or 'explain an atom'."
        animation_trigger = "chatbot_confused"

    return {
        "intent": context, # Pass back the new context
        "entities": {}, # LLM would extract more entities
        "response": response_text,
        "animation": animation_trigger,
        "doll_state_update": doll_state_update # Data for frontend to update dolls
    }

@app.route('/chat', methods=['POST'])
def chat():
    user_data = request.json
    user_message = user_data.get('user_message', '')
    context = user_data.get('context', None)
    # Get current doll state from frontend for context (or manage entirely backend)
    current_doll_state_from_frontend = user_data.get('doll_state', {}) 

    # For this demo, we use a global `current_atom_state` on backend for simplicity
    llm_output = call_llm_api(user_message, current_atom_state, context)

    # Update global state based on LLM's suggested changes
    if llm_output["doll_state_update"]:
        if "particle_add" in llm_output["doll_state_update"]:
            # This is already handled in the call_llm_api for simplicity
            pass
        if "reset_canvas" in llm_output["doll_state_update"] and llm_output["doll_state_update"]["reset_canvas"]:
            # Reset global state to initial for the target atom
            global current_atom_state
            current_atom_state = {
                "protons": 0, "neutrons": 0, "electrons": 0,
                "nucleus_stable": False, "electron_shells_stable": False,
                "target_atom": current_atom_state["target_atom"]
            }

    response_payload = {
        "text": llm_output["response"],
        "animation_trigger": llm_output.get("animation", None),
        "doll_state_update": llm_output["doll_state_update"], # Send doll updates to frontend
        "context": llm_output["intent"] # Update frontend context
    }
    return jsonify(response_payload)

if __name__ == '__main__':
    app.run(debug=True, port=5000)