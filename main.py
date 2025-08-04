from src.data.data_preparation import get_data
from src.retriever.retriever import Retriever


if __name__ == '__main__':
    
    documents = get_data()
    
    retriever = Retriever(document_corpus=documents)
    
    query1 = 'In which movie Tony Stark killed Thanos with Avengers?'
    query2 = 'In which movie Peter Parker is the main person?'
    query3 = 'In which movie Peter Parker villain is Lizard?'
    query4 = 'In which movie was a big ship and love story between Jack and Rose?'
    relevant_documents = retriever.retrieve(query=query2)
    
    print(relevant_documents)
    
    for doc in relevant_documents:
        print(documents[doc])
