# RAG-Movies
🎬 Ask it like a fan, get answers like a pro. This tiny RAG system lets you describe a movie in plain English and fetches the title you're thinking of.


## Disclaimer

To run this code properly, you must have access to the Hugging Face model 'meta-llama/Llama-3.2-3B-Instruct' and create a **.env** file containing your Hugging Face token:
```bash
HF_TOKEN = '<your_token>'
```


## Usage example

1. Terminal app:

- Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Install dependencies:
```bash
uv sync
```

- Run the code:
```bash
uv run app/main.py
```

2. Dockerized web app:

```bash
source deploy.sh
```

After starting the app, you can test it with a simple POST request:

```bash
curl -X POST http://localhost:8087/query   -H "Content-Type: application/json"   -d '{"query": "In which movie Peter Parker villain is Lizard Curt Connors?"}'
```