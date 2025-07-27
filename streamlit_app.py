from RAG_ChatBot import ChatBot
import streamlit as st

# Instantiate and cache the bot
@st.cache_resource(show_spinner=False)
def get_bot():
    return ChatBot()

bot = get_bot()

st.title("J.A.C.K.S.O.N")

# Initialize chat history
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
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.spinner("One moment please…"):
        answer = bot.rag_chain.run(user_prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
