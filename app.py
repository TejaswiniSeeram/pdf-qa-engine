import streamlit as st
import tempfile
import os
from rag import process_pdf, answer_question

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Engine",
    page_icon="📄",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("📄 PDF Question Answering Engine")
st.caption("Upload any PDF — ask anything about its content. Answers come only from your document.")

st.divider()

# ── Session state — remembers if PDF was processed ──────────
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# ── File upload ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf",
    help="Supported: any text-based PDF"
)

if uploaded_file:
    # Save to temp file so fitz can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Process only if this is a new upload
    if not st.session_state.pdf_processed:
        with st.spinner("Reading and indexing your PDF..."):
            try:
                num_chunks = process_pdf(tmp_path)
                st.session_state.pdf_processed = True
                st.session_state.num_chunks = num_chunks
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # Clean up temp file
    os.unlink(tmp_path)

    # Show success message
    if st.session_state.pdf_processed:
        st.success(f"✅ PDF indexed — {st.session_state.num_chunks} chunks created")

        st.divider()

        # ── Question input ───────────────────────────────────
        st.subheader("Ask a Question")
        query = st.text_input(
            "Type your question here",
            placeholder="e.g. What is the main topic of this document?"
        )

        if query:
            with st.spinner("Searching document..."):
                try:
                    answer, source_chunks = answer_question(query)

                    # ── Answer ───────────────────────────────
                    st.divider()
                    st.subheader("Answer")
                    st.write(answer)

                    # ── Sources ──────────────────────────────
                    st.divider()
                    with st.expander("📚 View source chunks used to generate this answer"):
                        for i, chunk in enumerate(source_chunks):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(chunk)
                            st.write("")

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

# ── Empty state ──────────────────────────────────────────────
else:
    st.session_state.pdf_processed = False
    st.info("👆 Upload a PDF above to get started")

    st.divider()
    st.markdown("#### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Upload**\nAny text-based PDF")
    with col2:
        st.markdown("**2. Index**\nSplit into chunks + embeddings")
    with col3:
        st.markdown("**3. Ask**\nType any question")
    with col4:
        st.markdown("**4. Answer**\nGrounded in your document only")

