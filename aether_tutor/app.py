from flask import Flask, request, jsonify, render_template
import json
import time
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Global State (for demonstration - use a DB in production) ---
# Stores student-specific data
student_data = {
    "default_student": { 
        "name": "Student",
        "learning_history": [],  
        "current_topic": None,
        "quiz_in_progress": False,
        "quiz_start_time": None,
        "quiz_questions": [],
        "quiz_score": 0,
        "study_timer_active": False,
        "study_timer_end": None,
        "last_interaction_time": datetime.now(),
        "registered_phone_number": None,
        "scheduled_reminders": [] # New: To store scheduled reminders
    }
}

# Knowledge Base (Simplified)
knowledge_base = {
    "atoms": {
        "explanation": "An atom is the basic building block of all matter. It consists of a central nucleus (made of protons and neutrons) surrounded by electrons. It's like a tiny solar system!",
        "interactive_prompt": "Would you like to try building a simple atom with our interactive builder?",
        "quiz_questions": [
            {"q": "What two particles are found in the nucleus of an atom?", "a": ["protons and neutrons"], "options": ["protons and electrons", "neutrons and electrons", "protons and neutrons"]},
            {"q": "Which particle orbits the nucleus?", "a": ["electron"], "options": ["proton", "neutron", "electron"]},
            {"q": "A neutral atom has an equal number of which two particles?", "a": ["protons and electrons"], "options": ["protons and neutrons", "neutrons and electrons", "protons and electrons"]}
        ]
    },
    "photosynthesis": {
        "explanation": "Photosynthesis is the process plants use to convert light energy into chemical energy, creating glucose (food) from carbon dioxide and water. It's how plants eat!",
        "interactive_prompt": "We can visualize the process with a simple diagram. Ready?",
        "quiz_questions": [
            {"q": "What gas do plants absorb for photosynthesis?", "a": ["carbon dioxide"], "options": ["oxygen", "carbon dioxide", "nitrogen"]},
            {"q": "What energy source do plants use in photosynthesis?", "a": ["sunlight"], "options": ["heat", "sunlight", "wind"]},
            {"q": "What is the main product of photosynthesis that plants use as food?", "a": ["glucose"], "options": ["oxygen", "water", "glucose"]}
        ]
    },
    "gravity": {
        "explanation": "Gravity is a fundamental force of attraction between any two objects with mass. The more massive an object, the stronger its gravitational pull. It's why things fall down and planets orbit stars!",
        "interactive_prompt": "Let's drop some virtual objects to see gravity in action!",
        "quiz_questions": [
            {"q": "Who is famously associated with the discovery of gravity?", "a": ["Isaac Newton"], "options": ["Albert Einstein", "Isaac Newton", "Galileo Galilei"]},
            {"q": "What causes objects to fall towards the Earth?", "a": ["gravity"], "options": ["magnetism", "air pressure", "gravity"]}
        ]
    },
    "python": { 
        "explanation": "Python is a popular high-level, interpreted programming language known for its readability and versatility. It's used for web development, data analysis, AI, and more. It's a great language to learn for beginners!",
        "interactive_prompt": "Would you like a small coding challenge in Python?",
        "quiz_questions": [
            {"q": "Which keyword is used to define a function in Python?", "a": ["def"], "options": ["func", "define", "def"]},
            {"q": "What is the output of 'print(2 + 3)' in Python?", "a": ["5"], "options": ["2+3", "5", "error"]}
        ]
    }
}

# --- Mock LLM Call (Crucial for AI-like behavior, replace with actual LLM API integration) ---
def call_llm_api(prompt, student_id="default_student", context=None):
    student = student_data[student_id]
    user_prompt_lower = prompt.lower()
    
    response_text = "I'm not sure how to respond to that. Could you ask me something about a concept, or perhaps start a timer?"
    animation_trigger = "chatbot_confused"
    action_trigger = None 
    interactive_data = {} 

    # --- Check for Due Reminders First ---
    # This simulates a background check, triggering notifications on next user interaction
    reminders_to_trigger = []
    for reminder in student["scheduled_reminders"]:
        if not reminder["triggered"] and datetime.now() >= reminder["time"]:
            reminders_to_trigger.append(reminder)
            reminder["triggered"] = True # Mark as triggered

    if reminders_to_trigger:
        # Prioritize reminder notification
        first_reminder = reminders_to_trigger[0]
        response_text = f"Aether: Just a reminder: {first_reminder['message']}!"
        animation_trigger = "phone_notification_animation" # Use phone animation for reminders
        if student["registered_phone_number"]:
            action_trigger = {"type": "send_mobile_notification", "title": "Aether Reminder", "body": first_reminder['message']}
        # Remove triggered reminders from the list (or keep for history)
        student["scheduled_reminders"] = [r for r in student["scheduled_reminders"] if not r["triggered"]]
        return { # Return immediately after triggering reminder
            "text": response_text,
            "animation_trigger": animation_trigger,
            "action_trigger": action_trigger,
            "interactive_data": interactive_data, 
            "context": context 
        }


    # --- Remembering History & Contextual Responses ---
    if "hello" in user_prompt_lower or "hi" in user_prompt_lower or "hey" in user_prompt_lower:
        if (datetime.now() - student["last_interaction_time"]).total_seconds() < 3600: 
            response_text = f"Welcome back, {student['name']}! Ready to continue learning?"
            if student["learning_history"]:
                response_text += f" Last time, we discussed: {', '.join(student['learning_history'][-2:])}." 
        else:
            response_text = f"Hello, {student['name']}! I'm Aether, your AI Tutor. How can I help you learn today?"
        animation_trigger = "waving_hand_greet"
        student["last_interaction_time"] = datetime.now()

    # --- Mobile Notification Registration ---
    elif "register phone" in user_prompt_lower or "set up notifications" in user_prompt_lower:
        action_trigger = {"type": "prompt_phone_number_input"}
        response_text = "Certainly! To set up notifications, please provide your mobile number. I'll use it to send you reminders or updates."
        animation_trigger = "phone_notification_animation" 
        context = "registering_phone_number" 
    
    elif context == "registering_phone_number" and any(char.isdigit() for char in user_prompt_lower):
        phone_number = ''.join(filter(str.isdigit, user_prompt_lower)) 
        if len(phone_number) >= 10: 
            student["registered_phone_number"] = phone_number
            response_text = f"Great! Your number {phone_number} has been registered for notifications. I'll send you a reminder when your study timer ends or quiz finishes!"
            animation_trigger = "correct_answer_sparkle" 
            context = None 
        else:
            response_text = "That doesn't look like a valid phone number. Please try again, including your country code if necessary."
            animation_trigger = "chatbot_confused"
            context = "registering_phone_number" 
    
    # --- Set Time-Based Reminder ---
    elif "remind me at" in user_prompt_lower or "set a reminder for" in user_prompt_lower:
        try:
            # Attempt to parse time (e.g., "11:30pm", "9 am", "14:00")
            time_str_match = None
            for pattern in [r'(\d{1,2}(:\d{2})?\s*(am|pm)?)', r'(\d{1,2}:\d{2})']:
                match = random.search(pattern, user_prompt_lower) # Using random.search instead of re.search
                if match:
                    time_str_match = match.group(0)
                    break

            if time_str_match:
                # Basic time parsing - needs robust error handling for real app
                # For simplicity, assume current date, parse time
                now = datetime.now()
                parsed_time = None
                
                # Try parsing with am/pm
                try:
                    parsed_time = datetime.strptime(time_str_match.replace(" ", ""), "%I:%M%p")
                except ValueError:
                    try:
                        parsed_time = datetime.strptime(time_str_match.replace(" ", ""), "%I%p")
                    except ValueError:
                        # Try parsing 24-hour format
                        try:
                            parsed_time = datetime.strptime(time_str_match, "%H:%M")
                        except ValueError:
                            pass # Fall through if no format matches

                if parsed_time:
                    # Combine with today's date
                    reminder_datetime = now.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0)
                    
                    # If the time is already past today, schedule for tomorrow
                    if reminder_datetime < now:
                        reminder_datetime += timedelta(days=1)

                    # Extract the reminder message
                    reminder_message = "your scheduled task"
                    if "to " in user_prompt_lower:
                        reminder_message_part = user_prompt_lower.split("to ", 1)[1].strip()
                        if reminder_message_part:
                            reminder_message = reminder_message_part

                    student["scheduled_reminders"].append({
                        "time": reminder_datetime,
                        "message": reminder_message,
                        "triggered": False
                    })
                    response_text = f"Okay, I've set a reminder for you for {reminder_datetime.strftime('%I:%M %p')} to '{reminder_message}'. I'll notify you!"
                    animation_trigger = "timer_start_animation"
                else:
                    response_text = "I couldn't understand the time you specified. Please try again, like 'remind me at 3:30 PM to study' or 'set a reminder for 14:00 to test'."
                    animation_trigger = "chatbot_confused"
        except Exception as e:
            print(f"Error parsing reminder: {e}")
            response_text = "I had trouble setting that reminder. Could you phrase it differently? (e.g., 'remind me at 3:30 PM to study')"
            animation_trigger = "chatbot_confused"

    # --- Study Timer Functionality ---
    elif "start study timer" in user_prompt_lower or ("study" in user_prompt_lower and "timer" in user_prompt_lower):
        duration_minutes = 25 
        for d in ["5", "10", "15", "20", "25", "30", "45", "60"]:
            if f"{d} minutes" in user_prompt_lower:
                duration_minutes = int(d)
                break
        student["study_timer_active"] = True
        student["study_timer_end"] = datetime.now() + timedelta(minutes=duration_minutes)
        response_text = f"Okay, I've started a {duration_minutes}-minute study timer. Focus time begins now! I'll let you know when it's done."
        animation_trigger = "timer_start_animation"
        action_trigger = {"type": "start_study_timer", "duration": duration_minutes * 60} 
    
    elif "stop study timer" in user_prompt_lower and student["study_timer_active"]:
        student["study_timer_active"] = False
        student["study_timer_end"] = None
        response_text = "Study timer stopped. Great work! What would you like to do next?"
        animation_trigger = "timer_stop_animation"
        action_trigger = {"type": "stop_study_timer"}
    
    elif "how much time" in user_prompt_lower and student["study_timer_active"]:
        if student["study_timer_end"]:
            remaining = student["study_timer_end"] - datetime.now()
            if remaining.total_seconds() > 0:
                mins, secs = divmod(int(remaining.total_seconds()), 60)
                response_text = f"You have {mins} minutes and {secs} seconds remaining on your study timer."
            else:
                response_text = "Your study timer has finished!"
                student["study_timer_active"] = False
                action_trigger = {"type": "stop_study_timer"} 
                if student["registered_phone_number"]:
                    action_trigger["send_mobile_notification"] = {"title": "Aether: Study Timer Done!", "body": "Your study session has ended. Great work!"}
        else:
            response_text = "No study timer is currently active."
        animation_trigger = "clock_animation"

    # --- Exam Timer / Quiz Functionality (Improved Recognition) ---
    elif any(phrase in user_prompt_lower for phrase in ["sample exam", "quiz me", "test me", "give me a test", "start a quiz"]):
        topic_found = None
        for topic in knowledge_base:
            if topic in user_prompt_lower:
                topic_found = topic
                break
        
        if topic_found and knowledge_base[topic_found]["quiz_questions"]:
            student["quiz_in_progress"] = True
            student["quiz_start_time"] = datetime.now()
            student["quiz_questions"] = random.sample(knowledge_base[topic_found]["quiz_questions"], min(3, len(knowledge_base[topic_found]["quiz_questions"])))
            student["quiz_score"] = 0
            student["current_topic"] = topic_found
            
            first_q = student["quiz_questions"][0]
            response_text = f"Alright, let's start a quick quiz on **{topic_found.capitalize()}**! You have 60 seconds per question. Question 1: {first_q['q']}<br>Options: {', '.join(first_q['options'])}"
            animation_trigger = "quiz_start_animation"
            action_trigger = {"type": "start_quiz_timer", "duration": 60, "total_questions": len(student["quiz_questions"])}
        else:
            response_text = "I can quiz you on topics like 'atoms', 'photosynthesis', 'gravity', or 'Python'. Which one?"
            animation_trigger = "chatbot_confused"

    elif student["quiz_in_progress"] and ("answer is" in user_prompt_lower or "my answer is" in user_prompt_lower or "it's" in user_prompt_lower):
        current_q_index = -1
        for i, q in enumerate(student["quiz_questions"]):
            if q is not None: 
                current_q_index = i
                break
        
        if current_q_index != -1:
            current_q = student["quiz_questions"][current_q_index]
            user_answer_text = user_prompt_lower.replace("answer is ", "").replace("my answer is ", "").replace("it's ", "").strip()
            
            is_correct = False
            for correct_option in current_q["a"]:
                if user_answer_text.lower() == correct_option.lower():
                    is_correct = True
                    break
            
            if is_correct:
                student["quiz_score"] += 1
                response_text = "That's correct! 🎉"
                animation_trigger = "correct_answer_sparkle"
            else:
                response_text = f"Not quite. The correct answer was: {', '.join(current_q['a'])}. Keep trying!"
                animation_trigger = "incorrect_answer_sad"
            
            student["quiz_questions"][current_q_index] = None 
            
            next_q_index = -1
            for i in range(len(student["quiz_questions"])):
                if student["quiz_questions"][i] is not None:
                    next_q_index = i
                    break

            if next_q_index != -1:
                next_q = student["quiz_questions"][next_q_index]
                response_text += f"<br><br>Next Question ({next_q_index + 1}/{len(knowledge_base[student['current_topic']]['quiz_questions'])}): {next_q['q']}<br>Options: {', '.join(next_q['options'])}"
                action_trigger = {"type": "reset_quiz_question_timer", "duration": 60} 
            else:
                response_text += f"<br><br>Quiz finished! You scored {student['quiz_score']} out of {len(knowledge_base[student['current_topic']]['quiz_questions'])}. Great effort!"
                student["learning_history"].append(f"Quiz on {student['current_topic']}: {student['quiz_score']}/{len(knowledge_base[student['current_topic']]['quiz_questions'])}")
                student["quiz_in_progress"] = False
                student["current_topic"] = None
                action_trigger = {"type": "stop_quiz_timer"}
                if student["registered_phone_number"]:
                    action_trigger["send_mobile_notification"] = {"title": "Aether: Quiz Complete!", "body": f"You scored {student['quiz_score']} out of {len(knowledge_base[student['current_topic']]['quiz_questions'])} on your {student['current_topic']} quiz."}
        else:
            response_text = "It seems the quiz has ended or I'm not sure which question you're answering."
            student["quiz_in_progress"] = False
            action_trigger = {"type": "stop_quiz_timer"}

    # --- Concept Explanation & Interactive Triggers ---
    elif "explain" in user_prompt_lower or "what is" in user_prompt_lower or "tell me about" in user_prompt_lower:
        topic_to_explain = None
        for topic, data in knowledge_base.items():
            if topic in user_prompt_lower:
                topic_to_explain = topic
                break
        
        if topic_to_explain:
            student["current_topic"] = topic_to_explain
            student["learning_history"].append(f"Learned about {topic_to_explain}")
            response_text = knowledge_base[topic_to_explain]["explanation"]
            animation_trigger = "explaining_concept_animation" 
            if "interactive_prompt" in knowledge_base[topic_to_explain]:
                response_text += f" {knowledge_base[topic_to_explain]['interactive_prompt']}"
                action_trigger = {"type": "prompt_interactive", "concept": topic_to_explain}
        else:
            # General fallback for unlisted topics - simulates LLM understanding
            response_text = f"That's an interesting topic! While I don't have a specific interactive for '{user_prompt_lower}', I can try to explain it in simpler terms, or perhaps suggest a related concept I know more about?"
            animation_trigger = "chatbot_thinking"

    # --- Handling Interactive Responses (from canvas) ---
    elif context == "interactive_atom_builder":
        action = request.json.get("action", "")
        particle_type = request.json.get("particle_type", "")
        
        if action == "add_particle":
            response_text = f"You've placed a {particle_type}! Keep building your atom."
            animation_trigger = f"{particle_type}_add_animation" 
            
            if random.random() < 0.2: 
                response_text = f"Fantastic! You've successfully built a basic atom! Great job!"
                animation_trigger = "atom_complete_sparkle"
                action_trigger = {"type": "end_interactive"}
                student["learning_history"].append(f"Completed Atom Builder")
        elif action == "reset_builder":
            response_text = "Atom builder reset. Let's try again!"
            animation_trigger = "reset_build_animation"
        else:
            response_text = "What would you like to do in the atom builder?"

    elif "thanks" in user_prompt_lower or "bye" in user_prompt_lower or "exit" in user_prompt_lower:
        response_text = "You're welcome! Keep up the great work. See you next time!"
        animation_trigger = "waving_hand_exit"
        action_trigger = {"type": "end_session"} 
        
    student_data[student_id]["last_interaction_time"] = datetime.now() 

    return {
        "text": response_text,
        "animation_trigger": animation_trigger,
        "action_trigger": action_trigger,
        "interactive_data": interactive_data, 
        "context": context 
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_data = request.json
    user_message = user_data.get('user_message', '')
    context = user_data.get('context', None)
    student_id = "default_student" 

    llm_output = call_llm_api(user_message, student_id, context)

    return jsonify(llm_output)

@app.route('/interactive_action', methods=['POST'])
def interactive_action():
    action_data = request.json
    student_id = "default_student" 
    
    simulated_llm_prompt = f"User performed interactive action: {action_data.get('action', 'unknown')} with particle {action_data.get('particle_type', 'none')}"
    
    context = "interactive_atom_builder" 
    
    llm_output = call_llm_api(simulated_llm_prompt, student_id, context)
    
    return jsonify(llm_output)


if __name__ == '__main__':
    app.run(debug=True, port=5000)