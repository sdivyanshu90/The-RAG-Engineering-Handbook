# %% [markdown]
# Source notebook: naive-rag.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %%
# Notebook cell 1
import warnings
warnings.filterwarnings("ignore")

# %%
# Notebook cell 2
get_ipython().system('pip install --upgrade pymilvus sentence-transformers huggingface-hub langchain_community langchain-text-splitters pymupdf tqdm')

# %%
# Notebook cell 3
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_API_KEY")

# %%
# Notebook cell 4
import os

os.environ["HF_TOKEN"] = hf_token

# %%
# Notebook cell 5
get_ipython().system('pip install pymilvus[milvus_lite]')

# %%
# Notebook cell 6
get_ipython().system('pip install langchain langchain-community pypdf pymupdf faiss-cpu sentence-transformers huggingface_hub')

# %%
# Notebook cell 7
# =============================
# IMPORTS
# =============================
import gradio as gr
import json
from tqdm import tqdm

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from huggingface_hub import InferenceClient


# =============================
# 1️⃣ LOAD PDF DATASET
# =============================
PDF_PATH = "/kaggle/input/os-by-silberschatz/Meta-Reports-Fourth-Quarter-and-Full-Year-2025-Results-2026.pdf"

loader = PyMuPDFLoader(PDF_PATH, extract_images=False)
docs = loader.load()

print(f"Loaded {len(docs)} pages")


# =============================
# 2️⃣ SPLIT INTO CHUNKS
# =============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(docs)
text_lines = [chunk.page_content for chunk in chunks]

print(f"Created {len(text_lines)} chunks")


# =============================
# 3️⃣ LOAD EMBEDDING MODEL
# =============================
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def emb_text(text):
    return embedding_model.encode(
        [text],
        normalize_embeddings=True
    ).tolist()[0]

# Get embedding dimension
embedding_dim = len(emb_text("test"))
print("Embedding dimension:", embedding_dim)


# =============================
# 4️⃣ SETUP MILVUS VECTOR DB
# =============================
milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collection_name = "rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",
    consistency_level="Strong",
)


# =============================
# 5️⃣ CREATE & INSERT EMBEDDINGS
# =============================
data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({
        "id": i,
        "vector": emb_text(line),
        "text": line
    })

milvus_client.insert(collection_name=collection_name, data=data)

print("Embeddings inserted into Milvus")


# =============================
# 6️⃣ LOAD LLM (LLAMA 3.1)
# =============================
repo_id = "meta-llama/Llama-3.1-8B-Instruct"
llm_client = InferenceClient(model=repo_id, timeout=120)


# =============================
# 7️⃣ PROMPT TEMPLATE
# =============================
PROMPT = """
Use the following pieces of information enclosed in <context> tags 
to answer the question enclosed in <question> tags.

If the answer is not in the context, say you don't know.

<context>
{context}
</context>

<question>
{question}
</question>
"""

# %%
# Notebook cell 8
# =============================
# 8️⃣ RAG FUNCTION FOR GRADIO
# =============================
def ask_question(question, history):

    # Embed user question
    question_embedding = emb_text(question)

    # Search in Milvus
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[question_embedding],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    # Extract retrieved chunks
    retrieved_chunks = [
        res["entity"]["text"] for res in search_res[0]
    ]

    context = "\n\n".join(retrieved_chunks)

    # Format prompt
    prompt = PROMPT.format(context=context, question=question)

    # Call LLM
    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    answer = response.choices[0].message.content.strip()

    return answer


# =============================
# 9️⃣ GRADIO UI
# =============================
demo = gr.ChatInterface(
    fn=ask_question,
    title="📘 RAG Assistant",
    description="Ask questions about the uploaded PDF dataset.",
)

demo.launch()

# %%
# Notebook cell 9

