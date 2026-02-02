"""RAG query functions: retrieve relevant chunks and generate answers."""

from mistralai import Mistral
import chromadb

from config import (
    MISTRAL_API_KEY,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    VECTORSTORE_DIR,
    TOP_K_RESULTS,
    TEMPERATURE
)


class MistralEmbeddingFunction:
    """Custom embedding function for ChromaDB using Mistral."""
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = EMBEDDING_MODEL
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not input:
            return []
        
        response = self.client.embeddings.create(
            model=self.model,
            inputs=input
        )
        return [item.embedding for item in response.data]


def get_collection():
    """Get the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    embedding_fn = MistralEmbeddingFunction(MISTRAL_API_KEY)
    
    return client.get_collection(
        name="tunnel_budget",
        embedding_function=embedding_fn
    )


def retrieve_chunks(query: str, collection, n_results: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieve relevant chunks for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return retrieved


def generate_answer(query: str, retrieved_chunks: list[dict], api_key: str) -> str:
    """Generate answer using Mistral Large with retrieved context."""
    
    client = Mistral(api_key=api_key)
    
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        
        # Format page info
        start_page = meta.get("start_page", "?")
        end_page = meta.get("end_page", start_page)
        if start_page == end_page:
            page_info = f"Page {start_page}"
        else:
            page_info = f"Pages {start_page}-{end_page}"
        
        # Include section if available
        section = meta.get("section", "")
        section_info = f" - {section}" if section else ""
        
        # Include table header context if available
        table_header = meta.get("table_header", "")
        header_info = f"\n[Table columns: {table_header}]" if table_header else ""
        
        context_parts.append(
            f"[Source {i}: {page_info}{section_info}]{header_info}\n{chunk['document']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # System prompt for RAG
    system_prompt = """You are a helpful assistant analyzing a construction infrastructure budget document.

Your task is to answer questions based ONLY on the provided context from the document.

Guidelines:
1. Always cite page numbers when referencing specific information (e.g., "According to page 45...")
2. If information comes from a table, mention the relevant context
3. If you cannot find the answer in the provided context, clearly state: "I couldn't find this information in the provided document sections."
4. Be precise with numbers and financial figures
5. If the context contains partial information, acknowledge what you found and what might be missing

Format your response clearly with:
- A direct answer to the question
- Supporting details from the document
- Page references for verification"""

    user_message = f"""Context from the construction budget document:

{context}

---

Question: {query}

Please provide a comprehensive answer based on the context above."""

    # Generate response
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=TEMPERATURE
    )
    
    return response.choices[0].message.content


def query_rag(query: str, n_results: int = TOP_K_RESULTS) -> dict:
    """Full RAG pipeline: retrieve + generate."""
    
    # Get collection
    collection = get_collection()
    
    # Retrieve relevant chunks
    retrieved = retrieve_chunks(query, collection, n_results)
    
    # Generate answer
    answer = generate_answer(query, retrieved, MISTRAL_API_KEY)
    
    return {
        "answer": answer,
        "sources": retrieved,
        "query": query
    }


# Example usage
if __name__ == "__main__":
    # Test query
    test_query = "What is the total budget for tunnel construction?"
    
    print(f"Query: {test_query}")
    print("-" * 50)
    
    result = query_rag(test_query)
    
    print("\nAnswer:")
    print(result["answer"])
    
    print("\n" + "-" * 50)
    print("Sources:")
    for i, source in enumerate(result["sources"], 1):
        meta = source["metadata"]
        print(f"\n[{i}] Page {meta.get('start_page', '?')} (distance: {source['distance']:.3f})")
        print(f"    Preview: {source['document'][:150]}...")
