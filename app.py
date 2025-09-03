# chatbot.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import re

app = FastAPI()

# Allow Storyline/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI key from Render env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Load PDF text (two PDFs supported) ===
PDF_PATH = os.environ.get("PDF_PATH", "yourfile.pdf")
PDF_PATH_2 = os.environ.get("PDF_PATH_2", "yourfile3.pdf")

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

# Combine both PDFs
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
    question = data.get("question", "").strip().lower()
    image_size = data.get("image_size", "512x512")  # default size

    if not question:
        return JSONResponse({"answer": "Please enter a question."})

    if not PDF_TEXT.strip():
        return JSONResponse({"answer": "I don't know about that."})

    # === Detect if user is asking for image ===
    if re.search(r"\b(generate|create|show|make)\b.*\b(image|picture|diagram|visual)\b", question):
        return await generate_image(question, image_size)

    # === Normal Q&A from PDF ===
    words = [w for w in question.split() if len(w) > 2]
    potential_contexts = []
    for para in PDF_TEXT.split("\n\n"):
        para_low = para.lower()
        score = sum(1 for w in words if w in para_low)
        if score > 0:
            potential_contexts.append((score, para))

    potential_contexts.sort(key=lambda x: x[0], reverse=True)
    relevant_context = "\n\n".join([p[1] for p in potential_contexts[:5]])

    if not relevant_context.strip():
        return JSONResponse({"answer": "I don't know about that."})

    try:
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant. Only answer using the provided PDF context. If the answer is not there, reply exactly: 'I don't know about that.'"
        }

        user_prompt = f"""
        Here is some context from the document(s):
        {relevant_context}

        Answer the following question strictly based on this context.
        If the answer is not in the context, reply exactly: "I don't know about that."

        User question: {question}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, {"role": "user", "content": user_prompt}],
            max_tokens=200
        )

        ai_answer = response.choices[0].message.content.strip()
        return JSONResponse({"answer": ai_answer})

    except Exception as e:
        print(f"[ERROR] OpenAI call failed: {e}")
        return JSONResponse({"answer": "Sorry, I couldn't process your request right now."})

# === Image generation function ===
async def generate_image(user_prompt: str, image_size: str):
    try:
        # Extract context for image from PDF
        words = [w for w in user_prompt.split() if len(w) > 2]
        potential_contexts = []
        for para in PDF_TEXT.split("\n\n"):
            para_low = para.lower()
            score = sum(1 for w in words if w in para_low)
            if score > 0:
                potential_contexts.append((score, para))

        potential_contexts.sort(key=lambda x: x[0], reverse=True)
        relevant_context = "\n\n".join([p[1] for p in potential_contexts[:3]])

        if not relevant_context.strip():
            return JSONResponse({"answer": "I don't know about that."})

        # Validate size
        if image_size not in ["256x256", "512x512", "1024x1024"]:
            image_size = "512x512"

        # Send request to OpenAI Image API
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=f"Generate an educational image strictly related to the following PDF context: {relevant_context}",
            size=image_size
        )

        image_url = img_resp.data[0].url
        return JSONResponse({
            "answer": f"Here is an image based on the PDF context (size: {image_size}):",
            "image_url": image_url,
            "size": image_size
        })

    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        return JSONResponse({"answer": "Sorry, I couldn't generate an image right now."})
