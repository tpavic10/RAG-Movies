import bm25s
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from src.retriever.embedder import SentenceEmbedder

class Retriever:
    
    def __init__(self, document_corpus: list[str]) -> None:
        self.documents = document_corpus
        self.bm25 = bm25s.BM25(corpus=self.documents)
        
        
    def retrieve(self, query: str, top_k: int = 5) -> list[int]:
        keyword_doc_indices = self.bm25_retrieve(query=query)
        semantic_search_doc_indices = self.semantic_search_retrieve(query=query)
        return self.reciprocal_rank_fusion(keyword_indices=keyword_doc_indices, semantic_indices=semantic_search_doc_indices, top_k=top_k)
        
        
    def bm25_retrieve(self, query: str, top_k: int = 10) -> list[int]:
        tokenized_data = bm25s.tokenize(self.documents)
        self.bm25.index(tokenized_data)
        
        tokenized_sample_query = bm25s.tokenize(query)
        results, _ = self.bm25.retrieve(tokenized_sample_query, k=top_k)

        return [self.documents.index(res) for res in results[0]]
    
    
    def semantic_search_retrieve(self, query: str, top_k: int = 10) -> list[int]:
        embedder = SentenceEmbedder()
        corpus_embeddings = embedder.get_embeddings(documents=self.documents)
        query_embedding = embedder.get_query_embedding(query=query)
    
        similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings) # need extra dim for query
        similarity_scores = similarity_scores.flatten()  # shape: (n_docs,)
        similarity_indices = np.argsort(similarity_scores)[::-1] 
        top_k_indices = similarity_indices[:top_k]

        return [int(x) for x in top_k_indices]
    
    def reciprocal_rank_fusion(self, keyword_indices: list[int], semantic_indices: list[int], top_k: int, k: int = 60, beta: int = 0.6) -> list[int]:
        rrf = {}
        betas = [beta, 1 - beta]
        indices = [semantic_indices, keyword_indices]
        
        for beta, ranked_list in zip(betas, indices):
            for rank, doc_id in enumerate(ranked_list, start=1):
                rrf[doc_id] = rrf.get(doc_id, 0) + (beta * (1 / (k + rank)))
                
        sorted_items = sorted(rrf, key=rrf.get, reverse=True)
        return sorted_items[:top_k]
        
        