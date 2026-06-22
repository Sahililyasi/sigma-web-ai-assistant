import streamlit as st
from process_incoming import ask_question

st.set_page_config(
    page_title="Sigma AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Sigma Web Development AI Assistant")
st.subheader("Ask questions from the Sigma Web Development Course")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
question = st.chat_input("Ask anything from the Sigma Web Development Course...")

if question:

    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # get answer
    with st.spinner("Thinking..."):
        answer = ask_question(question)

    # show assistant answer
    with st.chat_message("assistant"):
        st.markdown(answer["answer"])

        for source in answer["sources"]:
            st.info(
                f"📹 Video {source['number']} | "
                f"{source['title']} | "
                f"{source['start']:.2f}s - {source['end']:.2f}s"
            )


    st.session_state.messages.append(
        {"role": "assistant", "content": answer["answer"]}
    )