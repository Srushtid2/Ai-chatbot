import streamlit as st
import requests

st.title("💬 My Local Chatbot (Fast Mode)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show previous messages
for chat in st.session_state["messages"]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    st.session_state["messages"].append({"user": user_input, "bot": ""})

    # Stream response from Ollama
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": user_input, "stream": True},
            stream=True,
        )

        bot_reply = ""
        placeholder = st.empty()

        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response":"' in data:
                    text_piece = data.split('"response":"')[1].split('"')[0]
                    bot_reply += text_piece
                    placeholder.markdown(f"**Bot:** {bot_reply}")

        # Save reply
        st.session_state["messages"][-1]["bot"] = bot_reply
