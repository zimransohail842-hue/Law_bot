import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    """All Chroma-specific code lives here, so swapping to pgvector later means rewriting only this class."""

    def __init__(self, path="./chroma_db", collection="pakistan_constitution",
                 model="BAAI/bge-small-en-v1.5"):
        self.client = chromadb.PersistentClient(path=path)
        self.name = collection
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model, normalize_embeddings=True)
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            name=self.name, embedding_function=self.ef, metadata={"hnsw:space": "cosine"})

    def reset(self):
        """Drop and recreate, so chunks whose ids changed don't linger after re-chunking."""
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self._open()

    def upsert(self, chunks, batch_size=64):
        for i in range(0, len(chunks), batch_size):
            b = chunks[i:i + batch_size]
            self.collection.upsert(ids=[c.id for c in b],
                                   documents=[c.text for c in b],
                                   metadatas=[c.metadata for c in b])

    def query(self, text, n_results=5, where=None):
        return self.collection.query(query_texts=[text], n_results=n_results, where=where)

    def get_article(self, number):
        return self.collection.get(where={"article_number": str(number)})

