from src.data.data_preparation import get_data
from src.retriever.retriever import Retriever


def simple_evaluation() -> None:
    documents = get_data()
    retriever = Retriever(document_corpus=documents)
    
    query = 'In which movie Peter Parker is the main person?'
    gt_documents = [0, 90, 132, 144, 168, 170, 201, 1363]
    predicted_documents= retriever.retrieve(query=query, top_k=20)
    
    k_list = [5, 10, 15]
    
    for k in k_list:
        precision_k = precision_at_k(predicted_documents, gt_documents, k=k)
        recall_k = recall_at_k(predicted_documents, gt_documents, k=k)    

        print(f"Precision@{k}: {precision_k:.3f}")
        print(f"Recall@{k}: {recall_k:.3f}")
    
    
    
def precision_at_k(predicted: list[int], relevant: list[int], k: int) -> float:
    predicted_k = predicted[:k]
    true_positives = len(set(predicted_k) & set(relevant))
    return true_positives / k  # k is = tp + fp

def recall_at_k(predicted: list[int], relevant: list[int], k: int) -> float:
    predicted_k = predicted[:k]
    true_positives = len(set(predicted_k) & set(relevant))
    return true_positives / len(relevant) if relevant else 0.0