import io

import requests
import streamlit as st

st.set_page_config(
    page_title="Enterprise Knowledge Intelligence",
    page_icon="🧠",
    layout="wide",
)

DEFAULT_API_URL = "http://localhost:8000"
api_url = st.sidebar.text_input("Backend URL", DEFAULT_API_URL).rstrip("/")
st.sidebar.markdown("---")
st.sidebar.subheader("Knowledge base")

uploaded_files = st.sidebar.file_uploader(
    "Upload enterprise PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload policies, reports, proposals, research, and other business documents.",
)

if st.sidebar.button("Index selected documents", disabled=not uploaded_files, use_container_width=True):
    progress = st.sidebar.empty()
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        progress.info(f"Indexing {index}/{len(uploaded_files)}: {uploaded_file.name}")
        try:
            response = requests.post(
                f"{api_url}/documents/",
                files={
                    "file": (
                        uploaded_file.name,
                        io.BytesIO(uploaded_file.getvalue()),
                        "application/pdf",
                    )
                },
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            st.sidebar.success(
                f"{result['file']}: {result['chunks']} chunks indexed"
            )
        except requests.RequestException as exc:
            st.sidebar.error(f"Could not index {uploaded_file.name}: {exc}")
    progress.empty()

st.title("Enterprise Knowledge Intelligence")
st.caption(
    "Turn scattered business documents into searchable, explainable, evidence-backed intelligence."
)

try:
    kb_response = requests.get(f"{api_url}/documents/", timeout=20)
    kb_response.raise_for_status()
    kb_data = kb_response.json()
    documents = kb_data.get("documents", [])
    total_chunks = kb_data.get("total_chunks", 0)
    legacy_index = kb_data.get("legacy_index_present", False)
except requests.RequestException:
    documents = []
    total_chunks = 0
    legacy_index = False
    st.warning("The backend is not reachable. Start the FastAPI service and refresh this page.")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Indexed documents", len(documents))
metric2.metric("Indexed chunks", total_chunks)
metric3.metric("Knowledge base", "Legacy index" if legacy_index else ("Ready" if documents else "Empty"))

if legacy_index:
    st.info(
        "A previously created FAISS index was found, but its original document inventory "
        "is not available in the new registry. Clear and re-index documents to enable "
        "accurate inventory tracking."
    )

with st.expander("Manage indexed documents", expanded=False):
    if documents:
        st.dataframe(
            [
                {
                    "Document": item.get("file", "Unknown"),
                    "Chunks": item.get("chunks", 0),
                    "Indexed at": item.get("indexed_at", ""),
                }
                for item in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No registered documents yet. Upload PDFs using the sidebar.")

    st.warning("Clearing the knowledge base permanently deletes all indexed content and uploaded PDFs in this local application.")
    confirm_clear = st.checkbox("I understand. Clear the entire knowledge base.")
    if st.button(
        "Clear knowledge base",
        type="secondary",
        disabled=not confirm_clear,
    ):
        try:
            response = requests.delete(f"{api_url}/knowledge-base/", timeout=30)
            response.raise_for_status()
            st.success("Knowledge base cleared.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not clear the knowledge base: {exc}")

st.divider()
st.subheader("Analyse your enterprise knowledge")

analysis_mode = st.selectbox(
    "Choose an analysis workflow",
    [
        "Ask a question",
        "Executive summary",
        "Compare documents",
        "Identify risks and gaps",
        "Find conflicting information",
        "Generate recommendations",
    ],
)

mode_instructions = {
    "Ask a question": "Answer the question directly using only the retrieved document evidence.",
    "Executive summary": "Prepare an executive summary of the relevant information in the indexed documents. Organise it into key points, important facts, and implications. Use only retrieved evidence.",
    "Compare documents": "Compare the relevant documents on the dimensions requested. Highlight similarities, differences, and any areas where evidence is missing. Attribute findings to their source documents.",
    "Identify risks and gaps": "Identify risks, issues, assumptions, and information gaps supported by the retrieved evidence. For each item, explain the evidence and potential business implication. Do not invent risk ratings.",
    "Find conflicting information": "Look for statements in the retrieved evidence that appear inconsistent or contradictory. Quote or paraphrase both positions, identify their source documents, and explain whether the conflict is definite or may be due to different dates or scope.",
    "Generate recommendations": "Develop practical recommendations grounded in the retrieved evidence. For each recommendation, explain the supporting evidence, expected benefit, and any limitation. Clearly label suggestions as recommendations rather than facts.",
}

question = st.text_area(
    "Describe what you want to investigate",
    placeholder="Example: What are the main operational risks identified across these reports?",
    height=120,
)

if st.button("Run analysis", type="primary", disabled=not question.strip()):
    if not documents and not legacy_index:
        st.error("Upload and index at least one document before running an analysis.")
    else:
        with st.spinner("Retrieving evidence and preparing analysis..."):
            try:
                full_question = (
                    f"Analysis workflow: {analysis_mode}.\n"
                    f"Instructions: {mode_instructions[analysis_mode]}\n"
                    f"User request: {question.strip()}"
                )
                response = requests.post(
                    f"{api_url}/research/",
                    json={"question": full_question},
                    timeout=240,
                )
                response.raise_for_status()
                result = response.json()

                st.markdown("### Analysis")
                st.markdown(result["answer"])

                sources = result.get("sources", [])
                with st.expander(f"Evidence sources ({len(sources)})", expanded=True):
                    if sources:
                        for source in sources:
                            page = source.get("page")
                            page_label = f" · Page {page}" if page else ""
                            distance = source.get("retrieval_distance")
                            distance_label = (
                                f" · Retrieval distance: {distance}"
                                if distance is not None
                                else ""
                            )
                            st.markdown(
                                f"- **{source.get('file', 'Unknown source')}**"
                                f"{page_label} · Chunk {source.get('chunk_id', 'Unknown')}"
                                f"{distance_label}"
                            )
                    else:
                        st.write("No source metadata was returned.")
            except requests.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except ValueError:
                    detail = exc.response.text
                st.error(detail)
            except requests.RequestException as exc:
                st.error(f"Unable to reach the research service: {exc}")

st.divider()
st.caption(
    "Prototype note: outputs are generated from retrieved document passages. "
    "Review cited evidence before using findings in business decisions."
)
