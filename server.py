import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from groq import Groq

app = FastAPI(title="BISync AI API")

# Enable CORS for GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize ChromaDB client using existing pre-indexed collection
CHROMA_PATH = "chroma_db"
db_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db_client.get_or_create_collection(name="bis_standards")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_documents(req: QueryRequest):
    try:
        # Chroma retrieves matching context directly from local database vector index
        results = collection.query(query_texts=[req.query], n_results=3)
        context = "\n".join(results["documents"][0]) if results["documents"] and results["documents"][0] else ""

        # Groq generates response based on retrieved BIS standards context
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful compliance assistant for the Bureau of Indian Standards. Answer strictly based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuery: {req.query}"
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        answer = chat_completion.choices[0].message.content
        return {"response": answer, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "online", "message": "BISync AI Backend is running."}