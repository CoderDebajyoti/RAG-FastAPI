from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, data_path="Data/knowledge.txt", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts = self.load_texts(data_path)
        
        # Encode and store embeddings
        self.embeddings = self.model.encode(self.texts, convert_to_numpy=True)
        self.embeddings = np.array(self.embeddings, dtype="float32")

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def load_texts(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
        
    def retrieve(self, query, top_k=3):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = np.array(query_emb, dtype="float32")
        distances, indices = self.index.search(query_emb, top_k)
        return [self.texts[i] for i in indices[0]]
