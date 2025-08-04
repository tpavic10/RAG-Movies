import polars as pl

from src.data.config import DATASET_PATH

def get_data() -> list[dict]:
    df = pl.read_csv(DATASET_PATH)
    return [f"Title: {movie['Title']}\n Description: {movie['Overview']}\n Genre: {movie['Genre']}" for movie in df.to_dicts()]