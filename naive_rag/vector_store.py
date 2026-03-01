from pymilvus import MilvusClient
from tqdm import tqdm
from config import COLLECTION_NAME, MILVUS_DB_PATH
from embeddings import embed_text


def setup_milvus(dimension):
    client = MilvusClient(uri=MILVUS_DB_PATH)

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=dimension,
        metric_type="IP",
        consistency_level="Strong",
    )

    return client


def insert_embeddings(client, text_chunks):
    data = []

    for i, chunk in enumerate(tqdm(text_chunks, desc="Creating embeddings")):
        data.append({
            "id": i,
            "vector": embed_text(chunk),
            "text": chunk
        })

    client.insert(collection_name=COLLECTION_NAME, data=data)
    print("Embeddings inserted into Milvus")