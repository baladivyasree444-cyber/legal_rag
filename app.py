
import streamlit as st
from rag_pipeline import answer_question, chunks

# Page configuration
st.set_page_config(
    page_title="Legal Document RAG",
    page_icon="⚖️",
    layout="wide"
)

# Application title
st.title("⚖️ Legal Document Assistant")

st.write(
    "Ask questions about your uploaded legal agreements. "
    "The assistant retrieves relevant document sections "
    "and generates answers using Gemini."
)

st.info(
    "This is a learning project and does not provide "
    "professional legal advice."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get document types and IDs from existing chunks
document_types = sorted({
    chunk["document_type"]
    for chunk in chunks
    if chunk.get("document_type")
})

# Sidebar
with st.sidebar:

    st.header("About This Project")

    st.write("**Project:** Legal Document RAG")
    st.write("**Documents:** 8 legal agreements")
    st.write("**Embedding Model:** gemini-embedding-001")
    st.write("**Vector Database:** FAISS")
    st.write("**LLM:** Gemini")

    st.divider()

    # Document type filter
    st.subheader("Document Filters")

    selected_type = st.selectbox(
        "Select Document Type",
        options=["All"] + document_types
    )

    # Get document IDs based on selected document type
    if selected_type == "All":

        available_ids = sorted({
            chunk["document_id"]
            for chunk in chunks
            if chunk.get("document_id")
        })

    else:

        available_ids = sorted({
            chunk["document_id"]
            for chunk in chunks
            if chunk.get("document_type") == selected_type
        })

    # Document ID filter
    selected_id = st.selectbox(
        "Select Document ID",
        options=["All"] + available_ids
    )

    # Number of chunks to retrieve
    top_k = st.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.caption(
        "Answers are generated using retrieved document content."
    )

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("sources"):

                st.write("**Source Documents:**")

                for source in message["sources"]:
                    st.caption(f"📄 {source}")

            if message.get("retrieved_chunks"):

                with st.expander("View Retrieved Document Chunks"):

                    for i, chunk in enumerate(
                        message["retrieved_chunks"], start=1
                    ):

                        st.write(f"**Chunk {i}**")
                        st.caption(f"Source: {chunk['source']}")
                        st.write(chunk["text"])
                        st.divider()


# User question input
question = st.chat_input(
    "Ask a question about your legal documents..."
)

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Convert All selections to None
    document_type = (
        None if selected_type == "All" else selected_type
    )

    document_id = (
        None if selected_id == "All" else selected_id
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = answer_question(
                    question,
                    k=top_k,
                    document_type=document_type,
                    document_id=document_id
                )

                answer = result["answer"]
                sources = result["sources"]
                retrieved_chunks = result["retrieved_chunks"]

                st.markdown(answer)

                # Display sources
                if sources:

                    st.write("**Source Documents:**")

                    for source in sources:
                        st.caption(f"📄 {source}")

                # Display retrieved chunks
                if retrieved_chunks:

                    with st.expander(
                        "View Retrieved Document Chunks"
                    ):

                        for i, chunk in enumerate(
                            retrieved_chunks, start=1
                        ):

                            st.write(f"**Chunk {i}**")
                            st.caption(f"Source: {chunk['source']}")
                            st.write(chunk["text"])
                            st.divider()

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieved_chunks": retrieved_chunks
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
