from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split():
    loader = PyMuPDFLoader(PDF_PATH, extract_images=False)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)
    text_chunks = [chunk.page_content for chunk in chunks]

    print(f"Created {len(text_chunks)} chunks")

    return text_chunks