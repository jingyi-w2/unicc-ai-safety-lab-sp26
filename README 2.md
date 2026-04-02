# UNICC AI Safety Agent Demo

## Overview

This project demonstrates a multi judge AI safety evaluation system with a Streamlit frontend and a FastAPI backend.

The system:

1. Collects structured input about an AI agent
2. Evaluates it using multiple judges
3. Reconciles results through a critique step
4. Produces a final safety recommendation

## Project Structure

```text
UNICC CAPSTONE/
├── app/
├── demo.py
├── requirements.txt
├── HANDOFF_README.md
├── outputs/
├── logs/
└── artifacts/
```
## Setup
```text
pip install -r requirements.txt
```

## Run the Project
### 1. Start Backend
```text
cd ~/Desktop/"UNICC CAPSTONE"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Backend docs:
```text
http://127.0.0.1:8000/docs
```
### 2. Start Frontend
```text
cd ~/Desktop/"UNICC CAPSTONE"
streamlit run demo.py
```
Frontend:
```text
http://localhost:8501
```

### How It Works
- User submits agent information via the UI
- Frontend sends data to the backend API through POST /submit
- Backend runs three judges:
1. Technical risk
2. Governance and compliance
3. Operational and system risk
- Critique step reconciles outputs
- Synthesis step produces the final decision
## Example Input

### Deployment Context:
NYU AI Sandbox demo environment

## Agent Description:
A multi judge AI safety evaluation system that uses multiple expert judges and a synthesis step to assess the safety risk of an AI agent.

## Notes
-This is a simulated sandbox environment
-If judge modules fail, the system uses conservative fallback logic
-The system escalates to human review when uncertain
## Demo Summary

Input → API → Multi Judge Evaluation → Critique → Final Decision

## Run Instructions for Submission
1. Start backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
2. Start frontend
streamlit run demo.py
3. Open
http://127.0.0.1:8000/docs
http://localhost:8501
## Key Point

The interface acts as the entry point to a multi agent AI safety evaluation pipeline.
