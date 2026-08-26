import io
from pathlib import Path

import requests
import streamlit as st

API_URL = st.sidebar.text_input("Backend URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise AI Research Agent", page_icon="🔎", layout="wide")

st.title("Enterprise AI Research Agent")
st.caption("Research answers grounded in your enterprise knowledge base")

st.sidebar.header("Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.sidebar.button("Index Documents", disabled=not uploaded_files):
    for uploaded_file in uploaded_files:
        try:
            response = requests.post(
                f"{API_URL}/documents/",
                files={
                    "file": (
                        uploaded_file.name,
                        io.BytesIO(uploaded_file.getvalue()),
                        "application/pdf",
                    )
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            st.sidebar.success(
                f"{result['file']}: {result['chunks_added']} chunks indexed"
            )
        except requests.RequestException as exc:
            st.sidebar.error(f"Failed to index {uploaded_file.name}: {exc}")

st.subheader("Research Question")
question = st.text_area(
    "Ask a question about the indexed enterprise knowledge",
    placeholder="Example: What are the major opportunities for AI in manufacturing?",
    height=120,
)

if st.button("Research", type="primary", disabled=not question.strip()):
    try:
        response = requests.post(
            f"{API_URL}/research/",
            json={"question": question.strip()},
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()

        st.subheader("Research Answer")
        st.write(result["answer"])

    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", exc.response.text)
        st.error(detail)
    except requests.RequestException as exc:
        st.error(f"Unable to reach the research service: {exc}")
