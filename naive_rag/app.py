import gradio as gr
from dotenv import load_dotenv
from data_loader import load_and_split
from embeddings import get_embedding_dim
from vector_store import setup_milvus, insert_embeddings
from llm import load_llm
from rag_pipeline import ask_question
load_dotenv()

# Step 1: Load and split PDF
text_chunks = load_and_split()

# Step 2: Setup vector DB
embedding_dim = get_embedding_dim()
milvus_client = setup_milvus(embedding_dim)

# Step 3: Insert embeddings
insert_embeddings(milvus_client, text_chunks)

# Step 4: Load LLM
llm_client = load_llm()


# Step 5: Gradio wrapper
def chat_fn(message, history):
    return ask_question(message, milvus_client, llm_client)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="📘 RAG Assistant",
    description="Ask questions about the uploaded PDF dataset.",
)

if __name__ == "__main__":
    demo.launch(share = True)