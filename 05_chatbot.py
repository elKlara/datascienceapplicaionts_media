import streamlit as st
from ollama import chat, generate

if "my_messages" not in st.session_state:
    st.session_state["my_messages"] = []

st.title("Klaras wunderbarer Chatbot \u2728")

# Jedes Mal, wenn die App neu geladen wird, werden die vorherigen Nachrichten angezeigt
for message in st.session_state["my_messages"]:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Stelle mir eine Frage...")

if prompt:

    # Schritt 1: User Message verarbeiten
    user_message = {"role": "user", "content": prompt}
    st.session_state["my_messages"].append(user_message)
    st.chat_message("user").markdown(prompt)

    # Schritt 2: LLM-Response generieren
    response = chat(
        model="llama3.1", 
        messages=st.session_state["my_messages"]
        )
    answer = response["message"]["content"]

    # Schritt 3: LLM-Message verarbeiten
    ai_message = {"role": "ai", "content": answer}
    st.session_state["my_messages"].append(ai_message)
    st.chat_message("ai").markdown(answer)