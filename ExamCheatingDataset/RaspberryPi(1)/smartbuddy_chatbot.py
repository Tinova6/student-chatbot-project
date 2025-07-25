import google.generativeai as genai 
import time
genai.configure(api_key="AIzaSyCNzNirukUxw3fb3nbugIw15Bvez2SzrOds")
model=genai.GenerativeModel("gemini-pro")
chat=model.start_chat()
print("Hello smarat buddy! I'm chatbot")
def ask_chatbot(prompt):
    print(f"ChatBot:,{prompt}")
    try:
        response=model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR:{str(e)}"
def start_pomodoro():
    print("Pomodoro:Starting 25-minute session (demo:5 seconds)...")
    for i in range(5):
        print(f"{(i+1)*5} minutes passed... (demo)")
        time.sleep(1)
    print("Time for a 5-minute break!")
def add_note():
    note=input("Type your note:")
    with open("notes.txt","a")as f:
        f.write(note + "\n")
    print("Note saved successfully!")
print("Hello Buddy! I'm SmartBuddy Pi - your personal study friend")
while True:
    user_input=input("\nYou:")
    if user_input.lower() in ["exit","quit","bye"]:
        print("SmartBuddy:Bye Buddy! Keep learning and smiling")
        break
    elif "note" in user_input.lower():
        add_note()
    elif "timer" in user_input.lower() or "pomodoro" in user_input.lower():
        start_pomodoro()
    else:
        response=ask_chatbot(user_input)
        print("SmartBuddy:",response)