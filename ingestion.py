import os
import fitz  # PyMuPDF library 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

#initialize persistent DB & lightweight embedding model
client = chromadb.PersistentClient(path="./chroma_db")  #create a client that saves and loads the database from a local directory everytime
collection = client.get_or_create_collection(name="bis_standards")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = "./data"
documents, metadatas, ids = [], [], []

print("Extracting and parsing PDFs...")

 #process each PDF
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(DATA_DIR, filename)
        is_code = filename.replace(".pdf", "")
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            
            if not text:
                continue

            # 3. helps to split the text (even paeges into chunks)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600, 
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " "]
            )
            chunks = text_splitter.split_text(text)
            print(chunks)

            for idx, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "is_code": is_code,
                    "page_number": page_num + 1,
                    "source": filename
                })
                ids.append(f"{is_code}_p{page_num+1}_c{idx}")

#generate embeddings and save to ChromaDB vector database
print(f"Generating embeddings for {len(documents)} chunks...")
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("Ingestion complete! Local Chroma DB populated.")