def system_prompt() -> str:
    return """
    You are the movie title assistant. Answer the users query with a title of a movie in a friendly sentiment.
    There will be provided additional information about the movies for you to be more accurate in finding movie title.
    You should use your overall knowledge, but provided documents should help. If you can not give the movie name (title) answer "Unkown movie".
    """

    
def user_prompt(query: str, retrieved_context: str) -> str: 
    return f"""
    Context:\n
    {retrieved_context} \n\n
    User query:\n
    {query}
    """