import joblib
import torch
import pathlib
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from src.retriever.config import EMBEDDING_MODEL_NAME, EMBEDDINGS_DIR_PATH, EMBEDDER_BATCH_SIZE


class SentenceEmbedder:
    def __init__(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        self.embeddings_output_dir = pathlib.Path(EMBEDDINGS_DIR_PATH)
        
    def _save_embeddings_to_file(self, embeddings: np.ndarray) -> None:
        self.embeddings_output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(embeddings, self.embeddings_output_dir / 'embeddings.joblib')
        
    def _load_embeddings_from_file(self) -> np.ndarray:
        return joblib.load(self.embeddings_output_dir / 'embeddings.joblib')
        
    def create_corpus_embeddings(self, documents: list[str], batch_size: int = EMBEDDER_BATCH_SIZE) -> None:
        embeddings = []
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
        
        self._save_embeddings_to_file(np.array(embeddings))
        return np.array(embeddings)
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)
    
    def get_embeddings(self, documents: list[str]) -> np.ndarray:
        if (self.embeddings_output_dir / 'embeddings.joblib').exists():
            return self._load_embeddings_from_file()
        else:
            return self.create_corpus_embeddings(documents)
        
        