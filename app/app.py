from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.pipeline import run_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest) -> dict[str, str]:
    response = run_pipeline(request.query)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087, loop="uvloop")