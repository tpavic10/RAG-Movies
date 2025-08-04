def format_documents_to_string(documents: list[str], relevant_documents: list[int]) -> str:
    document_context = ""
    for i, doc in enumerate(relevant_documents, start=1):
        document_context += f"{i}. {documents[doc]}\n"
    return document_context