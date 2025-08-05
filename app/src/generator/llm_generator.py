import torch
from transformers import pipeline

from src.generator.config import MODEL_ID
from src.generator.prompts import system_prompt, user_prompt
from src.settings import Settings


class Generator:
    def __init__(self, device: str = 'cuda') -> None:
        self.pipeline = pipeline(
            "text-generation",
            model=MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
            token=Settings.HF_TOKEN
        )
        
    def generate(self, query: str, retrieved_context: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(query=query, retrieved_context=retrieved_context)},
        ]
        output = self.pipeline(messages, max_new_tokens=128)
        return output[0]["generated_text"][-1]['content']