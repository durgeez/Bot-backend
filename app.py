# chatbot.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import OpenAI
import os

app = FastAPI()

# Allow Storyline (browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # DEV: use "*" now; later restrict to your Storyline domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key from Render environment (not hardcoded!)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Load PDF text ===
PDF_PATH = os.environ.get("PDF_PATH", "yourfile.pdf")
PDF_PATH_2 = os.environ.get("PDF_PATH_2", "yourfile3.pdf")  # second PDF

def load_pdf_text(path):
    if not os.path.exists(path):
        print(f"[WARNING] PDF not found at: {path}")
        return ""
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append(txt.strip())
    return "\n\n".join(pages)

# Merge both PDFs into one big text
PDF_TEXT = load_pdf_text(PDF_PATH) + "\n\n" + load_pdf_text(PDF_PATH_2)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pdf1_loaded": bool(load_pdf_text(PDF_PATH)),
        "pdf2_loaded": bool(load_pdf_text(PDF_PATH_2))
    }

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()

    if not question:
        return JSONResponse({"answer": "Please enter a question."})

    if not PDF_TEXT.strip():
        return JSONResponse({"answer": "I don't know about that."})

    # === STEP 1: Find relevant context from PDFs using keyword match ===
    words = [w.lower() for w in question.split() if len(w) > 2]
    potential_contexts = []
    for para in PDF_TEXT.split("\n\n"):
        para_low = para.lower()
        score = sum(1 for w in words if w in para_low)
        if score > 0:
            potential_contexts.append((score, para))

    potential_contexts.sort(key=lambda x: x[0], reverse=True)
    relevant_context_paragraphs = [p[1] for p in potential_contexts[:5]]
    relevant_context = "\n\n".join(relevant_context_paragraphs)

    # If no relevant PDF content found → answer "I don't know about that."
    if not relevant_context.strip():
        return JSONResponse({"answer": "I don't know about that."})

    # === STEP 2: Use AI but ONLY with PDF context ===
    try:
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant. You must only answer using the provided PDF context. If the answer is not in the context, reply exactly: 'I don't know about that.'"
        }

        user_prompt = f"""
        Here is some context from the documents:
        {relevant_context}

        Answer the following question strictly based on this context.
        If the answer is not in the context, reply exactly: "I don't know about that."

        User question: {question}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                system_message,
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200
        )

        ai_answer = response.choices[0].message.content.strip()
        return JSONResponse({"answer": ai_answer})

    except Exception as e:
        print(f"[ERROR] OpenAI call failed: {e}")
        return JSONResponse({"answer": "Sorry, I couldn't process your request right now."})