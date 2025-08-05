from src.data.data_preparation import get_data
from src.retriever.retriever import Retriever
from src.generator.llm_generator import Generator
from src.utils import format_documents_to_string


def run_pipeline(query: str) -> str:
    
    documents = get_data()
    retriever = Retriever(document_corpus=documents)
    relevant_documents = retriever.retrieve(query=query, top_k=5)
    document_context = format_documents_to_string(documents=documents, relevant_documents=relevant_documents)
    
    generator = Generator()
    response = generator.generate(query, document_context)
    return response