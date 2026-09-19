# RAG Workflow

This document describes the end-to-end workflow implemented in the existing codebase.

Document
→ Ingestion
→ Text Extraction
→ Chunking
→ Metadata
→ Embeddings
→ FAISS Index
→ Query Embedding
→ Retrieval
→ Context Construction
→ Prompt
→ LLM
→ Response

## Workflow Details

1. Document ingestion
   - `DocumentLoader.load_all()` loads supported document types from the configured data directory.
   - Each document is normalized and assigned metadata, including `document_id`, `file_name`, `source`, and `document_type`.

2. Text extraction and normalization
   - `TextPreprocessor.clean_documents()` removes whitespace issues and strips invalid control characters.
   - Empty or unreadable content is filtered before indexing.

3. Chunking
   - `DocumentSplitter` uses `RecursiveCharacterTextSplitter` with a default chunk size of 500 and overlap of 50.
   - Each chunk receives `chunk_id`, `chunk_index`, and `chunk_total` metadata.

4. Embedding generation
   - `EmbeddingGenerator` reuses a lazy-loaded SentenceTransformer model.
   - Embeddings are generated in batches and validated for the expected dimension.

5. FAISS indexing
   - `VectorStore.create_index()` and `save_index()` persist the vector index to disk.
   - Metadata lives alongside the vector store in `index.pkl`.

6. Query embedding and retrieval
   - `Retriever.retrieve()` converts the query into an embedding and runs FAISS similarity search.
   - Top-k results are returned with metadata and similarity scores.

7. Context construction
   - `ContextBuilder.build()` merges the retrieved chunks into a formatted context string.
   - Duplicate text is removed and source information is preserved.

8. Prompting
   - `PromptBuilder` creates prompts with the required `SYSTEM INSTRUCTIONS`, `CONTEXT`, `USER QUESTION`, and `ANSWER` sections.

9. LLM response generation
   - `ResponseGenerator` sends the prompt to Ollama or HuggingFace and returns the answer.
   - The pipeline produces grounded answers and tracks latency.

10. Response output
   - The final result includes the generated answer, sources, retrieved chunks, model name, and latency summary.
