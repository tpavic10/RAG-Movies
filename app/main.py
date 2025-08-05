from src.pipeline import run_pipeline

if __name__ == "__main__":
    
    query1 = 'In which movie Tony Stark killed Thanos with Avengers?'
    query2 = 'In which movie Peter Parker is the main person?'
    query3 = 'In which movie Peter Parker villain is Lizard?'
    query4 = 'In which Harry Potter movie Severus Snape kills Dumbledore?'
    query5 = 'In which Marvel movies group of heroes fights againts enemies like aliens, Thanos, others.'
    
    query = input("Find a movie title: ")
    answer = run_pipeline(query)
    print("\nAnswer:", answer)