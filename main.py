from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
import re

app = FastAPI(title="Haven Backend", description="Neuro-Inclusive Safety Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ────────────────────────────────────────────────

class PrepRequest(BaseModel):
    mode: str          # "prepare" | "practice" | "understand" | "plan"
    situation: str
    details: Optional[str] = ""
    age: Optional[str] = ""

# ── System Prompts (neurodivergent-friendly, mirrors Lovable originals) ───────

SYSTEM_PROMPTS = {
    "prepare": """You are a calm, supportive assistant helping neurodivergent people prepare for situations.
Use simple, literal, concrete language. No metaphors. No sarcasm. No ambiguous phrases.
Break everything into small, numbered steps. Be specific and predictable.
Respond ONLY with valid JSON in this exact shape:
{
  "steps": ["step 1 text", "step 2 text", "step 3 text"],
  "sensoryTips": ["tip 1", "tip 2"],
  "whatToExpect": "A short, calm description of what will happen.",
  "exitPlan": "A simple sentence about how to leave or take a break if needed."
}
No markdown. No explanation outside the JSON. Only the JSON object.""",

    "practice": """You are a calm, supportive assistant helping neurodivergent people practice social situations through role-play.
Use simple, literal, concrete language. No metaphors. No sarcasm.
Respond ONLY with valid JSON in this exact shape:
{
  "scenario": "A short description of the practice scenario.",
  "yourRole": "What the user will say or do.",
  "theirRole": "What the other person might say or do.",
  "steps": ["step 1", "step 2", "step 3"],
  "encouragement": "A short, genuine, specific encouraging sentence."
}
No markdown. No explanation outside the JSON. Only the JSON object.""",

    "understand": """You are a calm, supportive assistant helping neurodivergent people understand why social situations happen.
Use simple, literal, concrete language. No metaphors. No sarcasm.
Respond ONLY with valid JSON in this exact shape:
{
  "explanation": "A clear, simple explanation of why this situation happens.",
  "commonReactions": ["reaction 1", "reaction 2", "reaction 3"],
  "whyItHappens": "A short factual explanation of the cause.",
  "whatItMeans": "What this situation usually means for the people involved.",
  "reminders": ["reminder 1", "reminder 2"]
}
No markdown. No explanation outside the JSON. Only the JSON object.""",

    "plan": """You are a calm, supportive assistant helping neurodivergent people make step-by-step plans.
Use simple, literal, concrete language. No metaphors. No sarcasm.
Respond ONLY with valid JSON in this exact shape:
{
  "goal": "Restate the goal in one clear sentence.",
  "steps": [
    { "step": 1, "action": "What to do", "detail": "More specific detail if needed" }
  ],
  "potentialChallenges": ["challenge 1", "challenge 2"],
  "successLooks": "A concrete description of what success looks like.",
  "backupPlan": "A simple alternative if the main plan doesn't work."
}
No markdown. No explanation outside the JSON. Only the JSON object.""",
}

# ── HuggingFace Inference API call ───────────────────────────────────────────

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Optional: set HF_TOKEN env var for higher rate limits (still free tier)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def build_prompt(mode: str, situation: str, details: str, age: str) -> str:
    age_context = f" The user is {age} years old." if age else ""
    details_context = f" Additional details: {details}" if details else ""
    return f"Situation: {situation}{details_context}{age_context}"

def extract_json(text: str) -> dict:
    """Extract JSON from model output, stripping any markdown fences."""
    # Remove markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in model response")

async def call_hf_inference(system_prompt: str, user_message: str) -> str:
    """Call HuggingFace free Inference API using chat format."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # Mistral instruct format
    prompt = f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.3,
            "return_full_text": False,
            "do_sample": True,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(HF_API_URL, headers=headers, json=payload)

    if resp.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading, please retry in 20 seconds.")
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment and try again.")
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Model API error: {resp.text}")

    data = resp.json()
    # HF returns list of generated_text
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", "")
    raise HTTPException(status_code=500, detail="Unexpected response format from model.")

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Haven backend is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/prepare")
async def prepare(req: PrepRequest):
    if req.mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{req.mode}'. Must be: prepare, practice, understand, plan.")
    if not req.situation or not req.situation.strip():
        raise HTTPException(status_code=400, detail="'situation' field is required and cannot be empty.")

    system_prompt = SYSTEM_PROMPTS[req.mode]
    user_message = build_prompt(req.mode, req.situation, req.details or "", req.age or "")

    raw_output = await call_hf_inference(system_prompt, user_message)

    try:
        result = extract_json(raw_output)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Could not parse model response as JSON. Raw: {raw_output[:300]}")

    return {"mode": req.mode, "result": result}
