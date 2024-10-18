import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained DialoGPT model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-medium"  # You can choose small, medium, or large
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to generate a response from the chatbot
def generate_response(user_input):
    # Encode the user input and append it to chat history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([st.session_state.chat_history, new_user_input_ids], dim=-1) if st.session_state.chat_history else new_user_input_ids

    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update chat history
    st.session_state.chat_history = chat_history_ids

    return response

# Streamlit UI
st.title("AI Chatbot - Ask Me About Artificial Intelligence")
st.write("Type your question about Artificial Intelligence below:")

# User input field
user_input = st.text_input("You:", "")

# Generate response on button click
if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        st.session_state.chat_history.append(tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt'))
        st.text_area("AI Chatbot:", value=response, height=200, max_chars=None, key=None)
    else:
        st.text_area("AI Chatbot:", value="Please ask a question!", height=200, max_chars=None, key=None)

