from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import Retriever
from generator import Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()
retriever = Retriever()
generator = Generator()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "RAG FastAPI is live! Use /docs to test the API"}

@app.post("/rag/")
def rag_response(request: QueryRequest):
    logging.info(f"Received query: {request.query}")
    try:
        context_passages = retriever.retrieve(request.query)
        logging.info(f"Retrieved context: {context_passages}")

        if not context_passages:
            raise HTTPException(status_code=404, detail="No context found for the given query.")

        context = "\n".join(context_passages)
        logging.info(f"Combined context: {context}")

        answer = generator.generate(request.query, context)
        logging.info(f"Generated answer: {answer}")

        return {
            "answer": answer,
            "context_used": context_passages
        }
    except Exception as e:
        logging.error(f"Error while processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
