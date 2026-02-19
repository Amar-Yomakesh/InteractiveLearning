import streamlit as st
import config
from utils import ChromaDBManager, RAGAssistant

st.set_page_config(page_title="CloudHub RAG Assistant", layout="centered")
st.title("CloudHub Documentation Assistant")
st.caption("Ask questions about MuleSoft CloudHub — powered by RAG")


@st.cache_resource
def load_rag():
    db_manager = ChromaDBManager(
        persist_directory=config.VECTOR_DB_PATH,
        collection_name=config.COLLECTION_NAME,
    )
    return RAGAssistant(db_manager)


rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source_title']}**  \n  {src['source_url']}")

if prompt := st.chat_input("Ask a question about CloudHub..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching docs & generating answer..."):
            result = rag.ask(
                question=prompt,
                chat_history=chat_history if chat_history else None,
            )
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for src in result["sources"]:
                    st.markdown(f"- **{src['source_title']}**  \n  {src['source_url']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
