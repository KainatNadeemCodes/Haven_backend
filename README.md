# Haven_backend
---
title: Haven Backend
emoji: 🏠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Haven Backend API

FastAPI backend for [Haven – Neuro-Inclusive Safety Platform](https://safespacehaven.vercel.app).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health check |
| POST | `/api/prepare` | AI-powered preparation assistant |

## POST `/api/prepare`

**Request body:**
```json
{
  "mode": "prepare | practice | understand | plan",
  "situation": "Going to a birthday party",
  "details": "It will be loud and crowded",
  "age": "12"
}
```

**Response:**
```json
{
  "mode": "prepare",
  "result": {
    "steps": ["..."],
    "sensoryTips": ["..."],
    "whatToExpect": "...",
    "exitPlan": "..."
  }
}
```

## Environment Variables (optional)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Your HuggingFace token (free) — increases rate limits |

## Powered by
- FastAPI + Uvicorn
- Mistral-7B-Instruct-v0.3 via HuggingFace Inference API
