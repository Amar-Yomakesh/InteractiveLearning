import os
import bcrypt
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CloudHub RAG Assistant", layout="centered")

# ---------------------------------------------------------------------------
# Authentication gate
# Credentials are read from st.secrets — never hardcoded here.
#
# Required secrets shape (in .streamlit/secrets.toml or Streamlit Cloud UI):
#   [auth]
#   username      = "your_username"
#   password_hash = "$2b$12$..."   # bcrypt hash — see secrets.toml.example
# ---------------------------------------------------------------------------

def _login_gate() -> None:
    """Block the app until the user authenticates. No-op once authenticated."""
    if st.session_state.get("authenticated"):
        return

    if "auth" not in st.secrets:
        st.error(
            "Authentication is not configured. "
            "Add an `[auth]` section to `.streamlit/secrets.toml`."
        )
        st.stop()

    st.title("CloudHub Documentation Assistant")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        expected_user = st.secrets["auth"]["username"]
        expected_hash = st.secrets["auth"]["password_hash"].encode()
        if username == expected_user and bcrypt.checkpw(password.encode(), expected_hash):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()


_login_gate()

# ---------------------------------------------------------------------------
# API key resolution
# Must happen before importing config/utils so the module-level value is right
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str | None:
    """st.secrets (Streamlit Cloud) → environment variable → None (UI prompt)."""
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY") or None


import config  # noqa: E402 — imported after env is potentially set
from utils import ChromaDBManager, RAGAssistant, CloudHubDocsScraper, DocumentChunker  # noqa: E402

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.title("CloudHub Documentation Assistant")
st.caption("Ask questions about MuleSoft CloudHub — powered by RAG")

# ---------------------------------------------------------------------------
# Sidebar: logout + API key input (shown only when key is absent from secrets/env)
# ---------------------------------------------------------------------------

with st.sidebar:
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

api_key = _resolve_api_key()

if not api_key:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Used only for this session. Never stored by this app.",
        )
        st.caption("[Get a key at platform.openai.com](https://platform.openai.com/api-keys)")

    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to get started.")
        st.stop()

# Propagate so ChromaDBManager (via config) picks up the correct value
os.environ["OPENAI_API_KEY"] = api_key
config.OPENAI_API_KEY = api_key  # override the module-level variable

# ---------------------------------------------------------------------------
# DB initialisation — cached for the lifetime of the server process
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db():
    return ChromaDBManager(
        persist_directory=config.VECTOR_DB_PATH,
        collection_name=config.COLLECTION_NAME,
    )


db = get_db()

# ---------------------------------------------------------------------------
# Auto-ingestion: runs once when the collection is empty (first cold start)
# On Streamlit Community Cloud the filesystem resets on each new deployment,
# so this triggers automatically after every deploy.
# ---------------------------------------------------------------------------

if db.collection.count() == 0:
    with st.status(
        "Building knowledge base — runs once and may take a few minutes…",
        expanded=True,
    ) as status:
        st.write(f"Scraping `{config.SCRAPE_BASE_URL}` (up to {config.SCRAPE_MAX_PAGES} pages)…")
        docs = CloudHubDocsScraper().scrape()

        st.write(f"Scraped **{len(docs)}** pages. Chunking…")
        chunks = DocumentChunker().chunk_documents(docs)

        st.write(f"Created **{len(chunks)}** chunks. Generating embeddings and storing…")
        db.add_chunks(chunks)

        status.update(
            label=f"Knowledge base ready — {db.collection.count()} chunks indexed.",
            state="complete",
        )
    st.rerun()

rag = RAGAssistant(db)

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

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
        with st.spinner("Searching docs & generating answer…"):
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
