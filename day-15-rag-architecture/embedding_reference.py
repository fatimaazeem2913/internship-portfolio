"""
embedding_reference.py
--------------------------
Correct, real sentence-transformers reference code -- for local use with
a real internet connection (huggingface.co is blocked in the verification
sandbox this project was built in; see verify_setup.py's docstring for
the full explanation of the TF-IDF substitution used there instead).

Run this locally to see genuine, real sentence-transformers embeddings
and real ChromaDB retrieval powered by them, end to end.
"""

from sentence_transformers import SentenceTransformer
import chromadb


def build_real_rag_index(corpus, model_name="all-MiniLM-L6-v2"):
    """
    Loads a real sentence-transformers model, encodes a corpus, and
    stores the resulting embeddings in a real ChromaDB collection.

    Returns (collection, model) so you can keep querying with the SAME
    model afterward -- using a DIFFERENT model to embed the query than
    the one used to embed the corpus would produce meaningless
    similarity scores, since different models place text in different,
    incompatible vector spaces.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus).tolist()

    client = chromadb.Client()
    collection = client.create_collection(name="rag_corpus")
    collection.add(
        documents=corpus,
        embeddings=embeddings,
        ids=[f"doc{i}" for i in range(len(corpus))],
    )
    return collection, model


def query_real_rag_index(collection, model, query, n_results=3):
    """Embeds a query with the SAME model used for the corpus, retrieves the top matches."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results["documents"][0]


if __name__ == "__main__":
    corpus = [
        "The transformer architecture uses self-attention to relate every token to every other token.",
        "Word2Vec learns one static embedding per word, computed once during training.",
        "BERT produces contextual embeddings that change based on surrounding words.",
        "Retrieval-Augmented Generation retrieves relevant documents before generating an answer.",
    ]

    collection, model = build_real_rag_index(corpus)

    query = "How do models handle word meaning that depends on context?"
    top_matches = query_real_rag_index(collection, model, query, n_results=2)

    print(f"Query: {query}\n")
    print("Top matches:")
    for i, doc in enumerate(top_matches, 1):
        print(f"  {i}. {doc}")
