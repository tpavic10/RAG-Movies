from src.data.data_preparation import get_data
from src.retriever.retriever import Retriever

THRESHOLD_MEAN_AVERAGE_PRECISION_AT_5 = 0.8
THRESHOLD_RECIPROCAL_RANK = 0.6

def test_deployment_metrics() -> None:
    documents = get_data()
    retriever = Retriever(document_corpus=documents)
    
    queries = [
        'In which movie Peter Parker is the main person?',
        'In which movies group of heroes fights againts enemies. It is Marvel production.'
    ]
    gt_documents = [
        [0, 90, 132, 144, 168, 170, 201, 1363],
        [138, 71, 171, 254]
    ]
    predictions = [retriever.retrieve(query=q, top_k=15) for q in queries]
    
    mean_average_precision_at_5 = mean_average_precision_at_k(predicted=predictions, relevant=gt_documents, k_range=5)
    reciprocal_rank_metric = mean_reciprocal_rank(predicted=predictions, relevant=gt_documents)
    
    assert mean_average_precision_at_5 >= THRESHOLD_MEAN_AVERAGE_PRECISION_AT_5
    assert reciprocal_rank_metric >= THRESHOLD_RECIPROCAL_RANK



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
    

def mean_reciprocal_rank(predicted: list[int], relevant: list[int]) -> float:
    mean_rr = 0.0
    for pred, gt in zip(predicted, relevant):
        for rank, doc_id in enumerate(pred, start=1):
            if doc_id in gt:
                mean_rr += (1.0 / rank)
                break
    return mean_rr / len(relevant)
    

def mean_average_precision_at_k(predicted: list[list[int]], relevant: list[list[int]], k_range: int) -> float:
    ## mean_average_precision is through more prompts
    mean_avg_precision = 0.0
    for pred, gt in zip(predicted, relevant):
        avg_precision = 0.0
        for k in range(1, k_range + 1):
            avg_precision += precision_at_k(predicted=pred, relevant=gt, k=k)
            
        mean_avg_precision += (avg_precision / k_range)
    return mean_avg_precision / len(relevant)
        

def precision_at_k(predicted: list[int], relevant: list[int], k: int) -> float:
    predicted_k = predicted[:k]
    true_positives = len(set(predicted_k) & set(relevant))
    return true_positives / k  # k is = tp + fp

def recall_at_k(predicted: list[int], relevant: list[int], k: int) -> float:
    predicted_k = predicted[:k]
    true_positives = len(set(predicted_k) & set(relevant))
    return true_positives / len(relevant) if relevant else 0.0