# streamlit_app.py
from RAG_ChatBot import ChatBot
import streamlit as st

# Instantiate the bot once and cache it
@st.cache_resource(show_spinner=False)
def get_bot():
    return ChatBot()

bot = get_bot()

st.title("J.A.C.K.S.O.N")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm J.A.C.K.S.O.N. How can I assist you?"}
    ]

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Accept user input
user_prompt = st.chat_input("Ask me anything…")
if user_prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.spinner("One moment please…"):
        answer = bot.rag_chain.run(user_prompt)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("One moment please…"):
            # Gemma returns a dict with key "result"
            answer = bot.rag_chain.run(user_prompt)   # or .invoke(user_prompt)["result"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
