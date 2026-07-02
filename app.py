import streamlit as st
import requests
from retriever import Retriever

st.title("💬 My Local Chatbot (Fast Mode)")

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for chat in st.session_state["messages"]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    st.session_state["messages"].append({"user": user_input, "bot": ""})

    relevant_chunks = retriever.retrieve(user_input, top_k=2)

    if relevant_chunks:
        context_text = "\n".join(relevant_chunks)
        final_prompt = (
            f"Use the following context if relevant to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_input}"
        )
    else:
        final_prompt = user_input

    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": final_prompt, "stream": True},
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

        st.session_state["messages"][-1]["bot"] = bot_reply

        if relevant_chunks:
            with st.expander("🔍 Retrieved context (RAG)"):
                for chunk in relevant_chunks:
                    st.markdown(f"- {chunk}")
