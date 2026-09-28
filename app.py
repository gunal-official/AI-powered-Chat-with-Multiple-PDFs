import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from utils import process_pdf, create_vectorstore, files_fingerprint

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with Multiple PDFs (RAG + Memory)")

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

# ---------------------------------------------------------------- session
defaults = {
    "messages": [],        # [{role, content, sources}]
    "vectorstore": None,
    "files_hash": None,    # fingerprint of the last processed upload
    "doc_names": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com. "
        "You can also set GROQ_API_KEY in a .env file.",
    )

    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    top_k = st.slider("Chunks to retrieve", 2, 10, 4)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.doc_names:
        st.divider()
        st.subheader("📚 Loaded documents")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")

# ---------------------------------------------------------------- upload
uploaded_files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    fingerprint = files_fingerprint(uploaded_files)

    # Only (re)build the vector store when the uploads actually changed —
    # Streamlit reruns this script on every interaction.
    if fingerprint != st.session_state.files_hash:
        all_docs = []
        with st.spinner("Processing PDFs..."):
            for file in uploaded_files:
                all_docs.extend(process_pdf(file.getvalue(), file.name))
            st.session_state.vectorstore = create_vectorstore(all_docs)

        st.session_state.files_hash = fingerprint
        st.session_state.doc_names = [f.name for f in uploaded_files]
        st.success(
            f"✅ {len(uploaded_files)} PDF(s) processed "
            f"({len(all_docs)} chunks). Ask questions below."
        )

# ---------------------------------------------------------------- history
def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['file']} — page {src['page']}**")
            st.caption(src["snippet"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        render_sources(msg.get("sources"))

# ---------------------------------------------------------------- chat
if prompt := st.chat_input("Ask something about your PDFs..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.vectorstore is None:
        response_text = "⚠️ Please upload PDFs first."
        with st.chat_message("assistant"):
            st.write(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
    elif not api_key:
        response_text = "🔑 Please enter your Groq API key in the sidebar."
        with st.chat_message("assistant"):
            st.write(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
        docs = retriever.invoke(prompt)

        context = "\n\n".join(
            f"[{d.metadata.get('source', '?')} — page {d.metadata.get('page', '?')}]\n"
            f"{d.page_content}"
            for d in docs
        )
        sources = [
            {
                "file": d.metadata.get("source", "?"),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:300] + "…",
            }
            for d in docs
        ]

        history = "".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
            for m in st.session_state.messages[:-1]
        )

        full_prompt = f"""You are an AI assistant that answers questions about the user's documents.

Use both the conversation history and the provided context.
If the answer is not in the context, say "I don't know".

Conversation History:
{history}

Context:
{context}

Question:
{prompt}
"""

        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )

        with st.chat_message("assistant"):
            try:
                # Stream tokens as they arrive instead of waiting
                response_text = st.write_stream(
                    chunk.content for chunk in llm.stream(full_prompt)
                )
            except Exception as e:
                response_text = f"❌ Error: {e}"
                st.write(response_text)
                sources = []
            render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "sources": sources,
            }
        )
