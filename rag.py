# Used for reading files and working with folders
import os

# Used to create a unique hash key for caching query embeddings
import hashlib

# SentenceTransformer is used to convert text into vector embeddings
from sentence_transformers import SentenceTransformer

# NumPy is used for handling vectors/arrays efficiently
import numpy as np

# FAISS is used for fast similarity search on embedding vectors
import faiss


# Pre-trained embedding model (lightweight + fast, good for RAG)
MODEL_NAME = "all-MiniLM-L6-v2"


class MiniRAG:
    def __init__(self, data_folder="data"):
        # Folder containing documents (.txt / .md) used as knowledge base
        self.data_folder = data_folder

        # Load the embedding model to convert text chunks into vectors
        self.model = SentenceTransformer(MODEL_NAME)

        # Stores chunks with metadata (chunk text + source filename)
        self.text_chunks = []   # list of dict: {"chunk": str, "source": str}

        # FAISS index object for similarity search
        self.index = None

        # Cache to store query embeddings (prevents re-encoding same query again)
        # Format: {hash_key: embedding_vector}
        self.query_embedding_cache = {}

    def load_docs(self):
        """
        Reads all .md and .txt files from the data folder.
        Returns: list of tuples -> (filename, file_content)
        """
        docs = []
        for filename in os.listdir(self.data_folder):
            # Only load markdown and text files
            if filename.endswith(".md") or filename.endswith(".txt"):
                path = os.path.join(self.data_folder, filename)

                # Read file content
                with open(path, "r", encoding="utf-8") as f:
                    docs.append((filename, f.read()))

        return docs

    def chunk_text(self, text, chunk_size=250):
        """
        Splits a document into smaller chunks of words.
        This improves retrieval quality and avoids embedding huge text blocks.
        """
        words = text.split()
        chunks = []

        # Create chunks of size 'chunk_size'
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))

        return chunks

    def build_index(self):
        """
        Loads documents, chunks them, generates embeddings,
        and builds a FAISS similarity search index.
        """
        docs = self.load_docs()

        # Convert each document into chunks and store source info
        for filename, content in docs:
            chunks = self.chunk_text(content)
            for c in chunks:
                self.text_chunks.append({
                    "chunk": c,
                    "source": filename
                })

        # Extract only the chunk text for embedding generation
        texts = [x["chunk"] for x in self.text_chunks]

        # Generate embeddings for all chunks
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Create FAISS index with correct embedding dimension
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        # Add embeddings into FAISS index (float32 required)
        self.index.add(embeddings.astype(np.float32))

    def _get_query_embedding(self, query: str):
        """
        Generates embedding for the query.
        Uses caching to avoid repeated computation for same query.
        """
        # Create a stable hash key for caching
        key = hashlib.md5(query.strip().lower().encode()).hexdigest()

        # Return cached embedding if available
        if key in self.query_embedding_cache:
            return self.query_embedding_cache[key]

        # Encode query into embedding vector
        emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)

        # Save embedding in cache for future use
        self.query_embedding_cache[key] = emb
        return emb

    def retrieve(self, query, top_k=3):
        """
        Retrieves top-k most relevant chunks for the given query.
        Returns: list of dict -> {"chunk": ..., "source": ...}
        """
        # Get query embedding (cached if already computed)
        query_emb = self._get_query_embedding(query)

        # Search FAISS index for top-k nearest embeddings
        distances, indices = self.index.search(query_emb, top_k)

        # Collect the retrieved chunks
        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])

        return results
