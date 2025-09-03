# chatbot.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import OpenAI
import os, re

app = FastAPI()

# Allow Storyline/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Keywords (Cybersecurity terms) ===
KEYWORDS = [
    "Cyber Security", "Basics of Cyber Security", "Confidentiality", "Integrity", "Availability", "CIA Triad",
    "Vulnerability", "Threats", "Attacks", "Active Attacks", "Passive Attacks", "Computer Criminals",
    "Web-based Attacks", "SQL Injection", "Code Injection", "XML Injection", "Log Injection", "DNS Spoofing",
    "Session Hijacking", "Phishing", "Brute Force", "Denial of Service", "DoS", "DDoS", "Dictionary Attack",
    "URL Interpretation", "File Inclusion Attack", "Man-in-the-Middle Attack", "System-based Attacks",
    "Virus", "Worm", "Trojan Horse", "Backdoor", "Bots", "Layers of Security", "Mission Critical Assets",
    "Data Security", "Application Security", "Endpoint Security", "Network Security", "Perimeter Security",
    "Human Layer", "Cyber Warfare", "Cyber Crime", "Cyber Terrorism", "Cyber Espionage", "Security Policy",
    "Virus and Spyware Protection Policy", "Firewall Policy", "Intrusion Prevention Policy",
    "Application and Device Control", "Cyberspace", "Information Technology Act 2000", "IT Act 2000",
    "Section 43", "Section 66", "Section 66B", "Section 66C", "Section 66D", "Indian Penal Code", "IPC",
    "Section 464", "Section 465", "Section 468", "Section 469", "Section 471", "Companies Act 2013",
    "NIST Cybersecurity Framework", "NCFS", "International Cyber Laws", "National Cyber Security Policy",
    "Vision", "Mission", "Objectives", "Cyber Forensics", "Digital Forensics", "Digital Evidence",
    "Email Forensics", "Header Analysis", "Bait Tactics", "Server Investigation", "Network Device Investigation",
    "Software Embedded Identifiers", "Sender Mailer Fingerprints", "MiTec Mail Viewer", "OST Viewer",
    "PST Viewer", "eMailTrackerPro", "EmailTracer", "Digital Forensics Lifecycle", "Collection",
    "Examination", "Analysis", "Reporting", "Forensics Investigation", "Technical Challenges",
    "Anti-forensics", "Steganography", "Cloud", "Legal Challenges", "Indian Evidence Act 1872",
    "Section 65B", "Resource Challenges", "Mobile Security", "Wireless Security", "Mobile Computing",
    "Portable Computer", "Tablet PC", "Internet Tablet", "PDA", "Personal Digital Assistant",
    "Smartphone", "Carputer", "Fly Fusion Pentop Computer", "3G", "4G", "Mobility Trends", "Skull Trojan",
    "Cabir Worm", "Mosquito Trojan", "Brador Trojan", "Lasco Worm", "Mobile Malware", "Denial-of-Service",
    "Overbilling Attack", "Spoofed PDP", "Signaling-level Attacks", "Credit Card Frauds", "M-Commerce",
    "Mobile Banking", "Registry Settings", "Authentication Service Security", "Mishing", "Vishing",
    "Smishing", "Bluetooth Hacking", "Organizational Security Policies", "Cost of Cyber Crimes",
    "IPR Issues", "Intellectual Property Rights", "Web Threats", "Security and Privacy", "Social Media Marketing",
    "Security Risks", "Social Computing", "Data Privacy Concepts", "Privacy Attacks", "Data Linking",
    "Profiling", "Privacy Policies", "Privacy Policy Languages", "Medical Privacy", "Financial Privacy",
    "Maharashtra Government Website Hack", "Indian Banks Financial Loss", "Parliament Attack",
    "Pune Nigerian Racket", "Email Spoofing", "Online Gambling", "Intellectual Property Crime", "Financial Fraud"
]

# === Load PDFs ===
PDF_PATH = os.environ.get("PDF_PATH", "yourfile.pdf")
PDF_PATH_2 = os.environ.get("PDF_PATH_2", "yourfile3.pdf")

def load_pdf_text(path):
    if not os.path.exists(path):
        return ""
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        if txt.strip():
            pages.append(txt.strip())
    return "\n\n".join(pages)

PDF_TEXT = load_pdf_text(PDF_PATH) + "\n\n" + load_pdf_text(PDF_PATH_2)

# === Context matching functions ===
def find_relevant_context_keywords(question: str, pdf_text: str, top_k: int = 5):
    q_low = question.lower()
    matched_keywords = [kw for kw in KEYWORDS if kw.lower() in q_low]
    potential_contexts = []
    for para in pdf_text.split("\n\n"):
        score = sum(1 for kw in matched_keywords if kw.lower() in para.lower())
        if score > 0:
            potential_contexts.append((score, para))
    potential_contexts.sort(key=lambda x: x[0], reverse=True)
    return [p[1] for p in potential_contexts[:top_k]]

def find_relevant_context_semantic(question: str, pdf_text: str, top_k: int = 5):
    words = [w for w in question.lower().split() if len(w) > 2]
    potential_contexts = []
    for para in pdf_text.split("\n\n"):
        score = sum(1 for w in words if w in para.lower())
        if score > 0:
            potential_contexts.append((score, para))
    potential_contexts.sort(key=lambda x: x[0], reverse=True)
    return [p[1] for p in potential_contexts[:top_k]]

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    image_size = data.get("image_size", "512x512")

    if not question:
        return JSONResponse({"answer": "Please enter a question."})
    if not PDF_TEXT.strip():
        return JSONResponse({"answer": "I don't know about that."})

    # Detect image request
    if re.search(r"\b(generate|create|show|make)\b.*\b(image|picture|diagram|visual)\b", question.lower()):
        return await generate_image(question, image_size)

    # Try keyword first, then semantic
    context = find_relevant_context_keywords(question, PDF_TEXT)
    if not context:
        context = find_relevant_context_semantic(question, PDF_TEXT)
    if not context:
        return JSONResponse({"answer": "I don't know about that."})

    relevant_context = "\n\n".join(context)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Only answer using the provided PDF context. If the answer is not there, reply exactly: 'I don't know about that.'"},
                {"role": "user", "content": f"Context from the PDF:\n{relevant_context}\n\nQuestion: {question}"}
            ],
            max_tokens=200
        )
        ai_answer = response.choices[0].message.content.strip()
        return JSONResponse({"answer": ai_answer})
    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        return JSONResponse({"answer": "Sorry, I couldn't process your request right now."})

# === Image generation (keyword + semantic fallback) ===
async def generate_image(user_prompt: str, image_size: str):
    context = find_relevant_context_keywords(user_prompt, PDF_TEXT, top_k=3)
    if not context:
        context = find_relevant_context_semantic(user_prompt, PDF_TEXT, top_k=3)
    if not context:
        return JSONResponse({"answer": "I don't know about that."})

    relevant_context = "\n\n".join(context)
    if image_size not in ["256x256", "512x512", "1024x1024"]:
        image_size = "512x512"

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=f"Generate an educational diagram strictly related to this cybersecurity PDF context:\n{relevant_context}\n\nUser request: {user_prompt}",
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
