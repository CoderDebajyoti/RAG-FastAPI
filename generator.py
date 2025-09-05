import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import google.generativeai as genai

# Load .env locally (ignored on Render where env vars are injected)
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class Generator:
    def __init__(self):
        # Initialize Gemini LLM via LangChain wrapper
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

    def generate(self, query: str, context: str) -> str:
        """Generate an answer based on query and context using Gemini."""
        prompt = f"""You are a helpful assistant. 
Use the provided context to answer the question accurately and concisely. 
If the answer is not in the context, say you don’t know.

Context:
{context}

Question:
{query}

Answer:"""

        try:
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"⚠️ Error generating response: {e}"
