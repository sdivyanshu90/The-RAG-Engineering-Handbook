from huggingface_hub import InferenceClient
from config import LLM_REPO_ID


def load_llm():
    return InferenceClient(model=LLM_REPO_ID, timeout=120)