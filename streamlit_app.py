import streamlit as st
import requests

st.title("LLM based RAG search")

message_placeholder=st.empty()
user_input=st.chat_input("Write a prompt.")

clear_cache_button = st.button("Clear Cache")

if clear_cache_button:
    with st.spinner('Clearing cache...'):
        try:
            response = requests.post('http://localhost:5000/clear_cache', timeout=10)
            response.raise_for_status()
            if response.status_code == 200:
                st.success("Cache cleared successfully.")
            else:
                st.error(f"API returned an error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

if user_input:
    with st.spinner('Generating response. Please wait...'):
        #send POST request to Flask API
        try:
            response = requests.post('http://localhost:5000/process', json={"input": user_input,"sites_required": 20}, timeout=250)
            response.raise_for_status()
            if response.status_code == 200:
                result = response.json()  #parse JSON response from Flask

                message_placeholder.empty()
                st.write("Result:")
                st.write(result.get("result", "No result returned"))
            else:
                st.error(f"API returned an error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
