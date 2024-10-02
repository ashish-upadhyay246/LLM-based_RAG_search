import streamlit as st
import requests

st.title("LLM based RAG search")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to display chat history
def display_chat_history():
    for chat in st.session_state.chat_history:
        st.markdown(st.write(f"**User:** {chat['user']}"))
        st.markdown(st.write(f"**Bot:** {chat['bot']}"))

# Clear Cache Button
clear_cache_button = st.button("Clear Cache")

if clear_cache_button:
    with st.spinner('Clearing cache...'):
        try:
            response = requests.post('http://localhost:5000/clear_cache', timeout=10)
            response.raise_for_status()
            if response.status_code == 200:
                st.success("Cache cleared successfully.")
                st.session_state.chat_history = []
            else:
                st.error(f"API returned an error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# User Input
user_input = st.chat_input("Write a prompt.")

if user_input:
    with st.spinner('Generating response. Please wait...'):
        # Send POST request to Flask API
        try:
            response = requests.post('http://localhost:5000/process', json={"input": user_input, "sites_required": 20}, timeout=250)
            response.raise_for_status()
            if response.status_code == 200:
                result = response.json()  # Parse JSON response from Flask

                # Store the query and response in the chat history
                st.session_state.chat_history.append({
                    'user': user_input,
                    'bot': result.get("result", "No result returned")
                })

            else:
                st.error(f"API returned an error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# Display chat history
display_chat_history()
