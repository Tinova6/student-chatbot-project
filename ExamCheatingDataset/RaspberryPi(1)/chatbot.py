import streamlit as st
import google.generativeai as genai

st.markdown("# Simple Chat Bot page ?")
st.sidebar.markdown("# Chat Bot page ?")

# Replace the google_api_key here
GOOGLE_API_KEY = "AIzaSyAYOcGGzRif9-OKBeP_ErHc8IMzky8_uoY" #Replace with Google_Api_Key 
genai.configure(api_key=GOOGLE_API_KEY)

## function to load Gemini Pro model and get repsonses
geminiModel=genai.GenerativeModel("gemini-pro") 
chat = geminiModel.start_chat(history=[])

def get_gemini_response(query):
    #Sends the conversation history with the added message and returns the model's response.
    instantResponse=chat.send_message(query,stream=True)
    return instantResponse

st.header("A simple Chat Bot")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

#Display a single-line text input widget.
inputText=st.text_input("Input: ",key="input")
submitButton=st.button("Get Instant answers")

if submitButton and inputText:
    # calls the get_gemini_response function by passing the inputText as query and gets the response as output
    output=get_gemini_response(inputText)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", inputText))
    st.subheader("The Response is")
    #Display the output in the app as Bot response
    for outputChunk in output:
        st.write(outputChunk.text)
        st.session_state['chat_history'].append(("Bot", outputChunk.text))
        
st.subheader("The Chat History is")
 # Piece of code to show the chat history in the app   
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")