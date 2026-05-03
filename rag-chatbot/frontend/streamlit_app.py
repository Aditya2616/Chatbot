import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

st.title("LLM-Powered RAG Chatbot")

api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

st.sidebar.subheader("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "PDF, TXT, DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if st.sidebar.button("Upload") and uploaded_files:
    files = [
        ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
        for file in uploaded_files
    ]
    response = requests.post(f"{api_url}/upload", files=files, timeout=120)
    if response.ok:
        st.sidebar.success(response.json().get("message", "Uploaded"))
    else:
        st.sidebar.error(response.text)

st.sidebar.divider()
if st.sidebar.button("Health check"):
    response = requests.get(f"{api_url}/health", timeout=30)
    if response.ok:
        st.sidebar.success(response.json())
    else:
        st.sidebar.error(response.text)

question = st.text_input("Ask a question")
if st.button("Send") and question:
    payload = {"question": question}
    response = requests.post(f"{api_url}/query", json=payload, timeout=120)
    if response.ok:
        result = response.json()
        st.subheader("Answer")
        st.write(result.get("answer"))
        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources")
            for source in sources:
                st.caption(f"Source: {source.get('source')} | Page: {source.get('page')}")
                st.write(source.get("snippet"))
    else:
        st.error(response.text)
