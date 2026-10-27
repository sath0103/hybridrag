
import html
from pathlib import Path

import networkx as nx
import streamlit as st

import backend


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 PDF Hybrid RAG Assistant")
st.caption(
    "Ask questions about your PDF using FAISS vector search "
    "and a NetworkX knowledge graph."
)


@st.cache_resource(show_spinner=False)
def get_resources():
    """Load existing indexes without re-running ingestion."""
    store = backend.load_vector_store()
    graph = backend.load_knowledge_graph()
    return store, graph


def get_all_chunks(store):
    """Read the persisted LangChain documents from FAISS."""
    documents = list(store.docstore._dict.values())
    documents.sort(
        key=lambda doc: (
            int(doc.metadata.get("page_number", 0)),
            int(str(doc.metadata.get("chunk_id", "chunk_0")).split("_")[-1])
            if str(doc.metadata.get("chunk_id", "chunk_0")).split("_")[-1].isdigit()
            else 0,
        )
    )
    return documents


def graph_to_dot(graph, max_nodes=50, max_edges=60):
    """Create a bounded Graphviz preview of the saved graph."""
    nodes = list(graph.nodes())[:max_nodes]
    visible = set(nodes)

    lines = [
        "digraph KnowledgeGraph {",
        'graph [rankdir="LR"];',
        'node [shape="box", style="rounded"];',
    ]

    for node in nodes:
        safe_node = html.escape(str(node), quote=True)
        lines.append(f'"{safe_node}";')

    edge_count = 0
    for source, target, attrs in graph.edges(data=True):
        if source not in visible or target not in visible:
            continue

        relation = html.escape(
            str(attrs.get("relation", "RELATED_TO")),
            quote=True,
        )
        safe_source = html.escape(str(source), quote=True)
        safe_target = html.escape(str(target), quote=True)

        lines.append(
            f'"{safe_source}" -> "{safe_target}" '
            f'[label="{relation}"];'
        )

        edge_count += 1
        if edge_count >= max_edges:
            break

    lines.append("}")
    return "\n".join(lines)


# Sidebar: ingestion and index status
with st.sidebar:
    st.header("Document management")

    default_pdf = DATA_DIR / "spotify_web_app_architecture.pdf"

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Uploading and ingesting a new PDF rebuilds the local indexes.",
    )

    ingest_clicked = st.button(
        "Ingest / Rebuild Indexes",
        type="primary",
        use_container_width=True,
    )

    if ingest_clicked:
        if uploaded_pdf is not None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            selected_pdf = DATA_DIR / Path(uploaded_pdf.name).name
            selected_pdf.write_bytes(uploaded_pdf.getvalue())
        else:
            selected_pdf = default_pdf

        if not selected_pdf.exists():
            st.error("PDF not found. Upload a PDF or check the data folder.")
        else:
            try:
                with st.spinner(
                    "Extracting PDF, building FAISS and extracting graph entities..."
                ):
                    result = backend.ingest_pdf(selected_pdf)

                get_resources.clear()
                st.session_state.pop("rag_resources_error", None)

                st.success(
                    f"Ingestion complete: {len(result['pages'])} pages, "
                    f"{len(result['chunks'])} chunks, "
                    f"{result['graph'].number_of_nodes()} graph nodes, "
                    f"{result['graph'].number_of_edges()} graph edges."
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Index status")

    try:
        store, graph = get_resources()
        st.success("FAISS index loaded")
        st.success("Knowledge graph loaded")
        st.metric("FAISS vectors", store.index.ntotal)
        st.metric("Graph nodes", graph.number_of_nodes())
        st.metric("Graph edges", graph.number_of_edges())
    except Exception as exc:
        store, graph = None, None
        st.session_state["rag_resources_error"] = str(exc)
        st.warning("Indexes are not ready.")
        st.caption(str(exc))


if store is None or graph is None:
    st.info(
        "Use the sidebar to ingest your PDF, or run backend.py first "
        "to create the FAISS index and knowledge graph."
    )
    st.stop()


tab_chat, tab_chunks, tab_vector, tab_graph = st.tabs(
    [
        "💬 Chat",
        "📄 PDF Chunks",
        "🔎 Vector Search",
        "🕸️ Knowledge Graph",
    ]
)


# Tab 1: Chatbot
with tab_chat:
    st.subheader("Ask your document")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

            if item.get("sources"):
                with st.expander("View source chunks"):
                    st.write(item["sources"])

    question = st.chat_input(
        "Ask about architecture, microservices, databases, security..."
    )

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching vectors and graph evidence..."):
                    result = backend.answer_question(
                        question,
                        store=store,
                        graph=graph,
                    )

                answer = result["answer"]
                st.markdown(answer)

                with st.expander("Vector retrieval results"):
                    for document, distance in result["vector_results"]:
                        st.markdown(
                            f"**{document.metadata.get('chunk_id', 'unknown')}** "
                            f"· Page {document.metadata.get('page_number', 'unknown')} "
                            f"· FAISS distance: {distance:.4f}"
                        )
                        st.write(document.page_content)

                with st.expander("Knowledge graph evidence"):
                    if result["graph_results"]:
                        for source, relation, target, chunk_id in result["graph_results"]:
                            st.markdown(
                                f"`{source}` — **{relation}** → `{target}` "
                                f"(source: `{chunk_id}`)"
                            )
                    else:
                        st.info(
                            "No matching graph entities were found for "
                            "the question terms."
                        )

                sources = result["source_chunks"]
                if sources:
                    st.caption("Source chunk IDs: " + ", ".join(sources))

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                st.error(f"Question processing failed: {exc}")


# Tab 2: Inspect every stored PDF chunk
with tab_chunks:
    st.subheader("Stored PDF chunks")

    chunks = get_all_chunks(store)
    st.write(f"Total stored chunks: **{len(chunks)}**")

    if chunks:
        chunk_labels = [
            (
                f"{doc.metadata.get('chunk_id', 'unknown')} | "
                f"Page {doc.metadata.get('page_number', 'unknown')}"
            )
            for doc in chunks
        ]

        selected_index = st.selectbox(
            "Select a chunk",
            options=range(len(chunks)),
            format_func=lambda index: chunk_labels[index],
        )

        selected_chunk = chunks[selected_index]

        st.json(
            {
                key: value
                for key, value in selected_chunk.metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
        )

        st.text_area(
            "Chunk text",
            value=selected_chunk.page_content,
            height=300,
            disabled=True,
        )

        with st.expander("Preview all chunk IDs"):
            st.dataframe(
                [
                    {
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "page_number": doc.metadata.get("page_number"),
                        "characters": len(doc.page_content),
                    }
                    for doc in chunks
                ],
                use_container_width=True,
            )


# Tab 3: Run vector similarity search independently
with tab_vector:
    st.subheader("FAISS vector search")

    vector_question = st.text_input(
        "Search the PDF",
        key="vector_question",
        placeholder="e.g. What does the Catalog Service do?",
    )

    top_k = st.slider("Number of results", min_value=1, max_value=10, value=4)

    if st.button("Search FAISS"):
        if not vector_question.strip():
            st.warning("Enter a question or search phrase.")
        else:
            try:
                results = backend.retrieve_vector_chunks(
                    vector_question,
                    store=store,
                    k=top_k,
                )

                for rank, (document, distance) in enumerate(results, start=1):
                    with st.expander(
                        f"Result {rank}: "
                        f"{document.metadata.get('chunk_id', 'unknown')} "
                        f"(distance {distance:.4f})",
                        expanded=(rank == 1),
                    ):
                        st.write(
                            f"Page: {document.metadata.get('page_number', 'unknown')}"
                        )
                        st.write(document.page_content)
                        st.json(
                            {
                                key: value
                                for key, value in document.metadata.items()
                                if isinstance(value, (str, int, float, bool))
                            }
                        )

            except Exception as exc:
                st.error(f"Vector search failed: {exc}")


# Tab 4: Inspect graph entities and relationships
with tab_graph:
    st.subheader("NetworkX knowledge graph")

    graph_col1, graph_col2 = st.columns(2)
    graph_col1.metric("Entities / nodes", graph.number_of_nodes())
    graph_col2.metric("Relationships / edges", graph.number_of_edges())

    if graph.number_of_nodes() == 0:
        st.warning("The knowledge graph is empty. Rebuild the indexes.")
    else:
        st.caption(
            "The visualization is a preview of the graph, limited to the "
            "first 50 nodes and 60 edges for readability."
        )
        st.graphviz_chart(graph_to_dot(graph), use_container_width=True)

        node_names = sorted(str(node) for node in graph.nodes())
        selected_node = st.selectbox("Inspect an entity", node_names)

        attrs = graph.nodes[selected_node]
        st.write("**Entity metadata**")
        st.json(
            {
                key: value
                for key, value in attrs.items()
                if isinstance(value, (str, int, float, bool))
            }
        )

        st.write("**Outgoing relationships**")
        outgoing = list(graph.out_edges(selected_node, data=True))
        if outgoing:
            st.dataframe(
                [
                    {
                        "source": source,
                        "relation": data.get("relation", "RELATED_TO"),
                        "target": target,
                        "source_chunk": data.get("source_chunk", ""),
                    }
                    for source, target, data in outgoing
                ],
                use_container_width=True,
            )
        else:
            st.info("This entity has no outgoing relationships.")

        st.write("**Incoming relationships**")
        incoming = list(graph.in_edges(selected_node, data=True))
        if incoming:
            st.dataframe(
                [
                    {
                        "source": source,
                        "relation": data.get("relation", "RELATED_TO"),
                        "target": target,
                        "source_chunk": data.get("source_chunk", ""),
                    }
                    for source, target, data in incoming
                ],
                use_container_width=True,
            )
        else:
            st.info("This entity has no incoming relationships.")
