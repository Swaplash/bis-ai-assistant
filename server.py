from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

app = FastAPI(title="BIS Assistant Prototype")

# CORS setup for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize local models and DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="bis_standards")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing!")

groq_client = Groq(api_key=api_key)

class QueryRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def search_standards(request: QueryRequest):
    # step 1. Vector Search
    query_vector = embedder.encode([request.message]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=3)

    chunks = results["documents"][0]
    meta = results["metadatas"][0]

    # step 2. Inject metadata directly into the text stream
    context_parts = []
    extracted_sources = []
    
    for i in range(len(chunks)):  
        is_code = meta[i]["is_code"]
        page = meta[i]["page_number"]
        
        # Tag the chunk so the LLM reads the exact source and page
        source_tag = f"--- [Document: {is_code}, Page: {page}] ---"
        context_parts.append(f"{source_tag}\n{chunks[i]}")
        
        # Save for the JSON payload footer
        extracted_sources.append({"is_code": is_code, "page": page})

    context_text = "\n\n".join(context_parts)

   
    prompt = f"""
    You are an expert BIS Compliance AI Assistant. 
    Answer the user's question using ONLY the provided official standard context below. 
    You must cite the exact Document and Page numbers provided in the context tags.
    Format your response with a Markdown table where applicable.

    Context:
    {context_text}

    User Question: {request.message}
    """

    #LLM generation
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise technical compliance assistant. You never hallucinate data outside the context."},
            {"role": "user", "content": prompt}
        ],
        model="openai/gpt-oss-20b", # Swapped to a standard reliable Groq model
        temperature=0.1 # Low temperature for factual consistency
    )
    
    return {
        "status": "success",
        "reply": chat_completion.choices[0].message.content,
        "matched_sources": extracted_sources
    }

