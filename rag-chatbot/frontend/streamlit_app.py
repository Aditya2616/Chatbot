import uuid
from typing import Dict, List

import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

DEFAULT_API_URL = "http://localhost:8000"


def get_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") or payload.get("message")
    except ValueError:
        detail = response.text
    return detail or "Request failed."


def show_response_error(response: requests.Response) -> None:
    st.error(get_error_detail(response))


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL


def normalize_api_url() -> None:
    st.session_state.api_url = st.session_state.api_url.rstrip("/")


def render_sources(sources: List[Dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            source_name = source.get("source") or "Unknown source"
            page = source.get("page")
            page_label = f" | Page: {page}" if page is not None else ""
            st.caption(f"Source: {source_name}{page_label}")
            snippet = source.get("snippet")
            if snippet:
                st.write(snippet)


init_state()
normalize_api_url()

st.title("LLM-Powered RAG Chatbot")
st.caption("Upload documents from the sidebar and chat with your knowledge base.")

st.sidebar.text_input("API URL", key="api_url", on_change=normalize_api_url)
api_url = st.session_state.api_url

st.sidebar.subheader("Chat settings")
top_k = st.sidebar.number_input("Top K results", min_value=1, max_value=20, value=4)
use_history = st.sidebar.checkbox("Use chat history", value=True)
return_sources = st.sidebar.checkbox("Return sources", value=True)

if st.sidebar.button("New chat"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.caption(f"Session ID: {st.session_state.session_id}")

st.sidebar.divider()
st.sidebar.subheader("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "PDF, TXT, DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if st.sidebar.button("Upload"):
    if not uploaded_files:
        st.sidebar.warning("Select at least one file to upload.")
    else:
        files = [
            ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
            for file in uploaded_files
        ]
        with st.spinner("Uploading documents..."):
            try:
                response = requests.post(f"{api_url}/upload", files=files, timeout=120)
            except requests.RequestException as exc:
                st.sidebar.error(str(exc))
            else:
                if response.ok:
                    st.sidebar.success(response.json().get("message", "Uploaded"))
                else:
                    show_response_error(response)

st.sidebar.divider()
if st.sidebar.button("Health check"):
    try:
        response = requests.get(f"{api_url}/health", timeout=30)
    except requests.RequestException as exc:
        st.sidebar.error(str(exc))
    else:
        if response.ok:
            st.sidebar.success(response.json())
        else:
            show_response_error(response)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("cached"):
                st.caption("Cached response")
            render_sources(message.get("sources", []))

prompt = st.chat_input("Ask a question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "question": prompt,
        "session_id": st.session_state.session_id,
        "top_k": top_k,
        "use_history": use_history,
        "return_sources": return_sources,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{api_url}/query", json=payload, timeout=120)
            except requests.RequestException as exc:
                error_message = str(exc)
                error_display = f"⚠️ {error_message}"
                st.markdown(error_display)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_display}
                )
            else:
                if response.ok:
                    result = response.json()
                    answer = result.get("answer") or "No response."
                    sources = result.get("sources") or []
                    cached = result.get("cached", False)
                    st.markdown(answer)
                    if cached:
                        st.caption("Cached response")
                    render_sources(sources)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "cached": cached,
                        }
                    )
                else:
                    error_detail = get_error_detail(response)
                    error_display = f"⚠️ {error_detail}"
                    st.markdown(error_display)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_display}
                    )
