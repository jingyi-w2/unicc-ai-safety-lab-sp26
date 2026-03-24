# UNICC AI Safety Lab (Spring 2026)

A council based AI Safety Lab for evaluating AI agents before deployment through three expert judges and one synthesis layer.

## Overview

This project aims to build a multi module AI Safety Lab for pre deployment evaluation of AI agents in an institutionally controlled environment. The system is designed to assess AI agents from multiple independent perspectives, compare expert outputs, surface disagreements, and generate transparent and auditable recommendations for human review.

The current repository contains the Project 1 platform foundation, including:
- a runnable local prototype pipeline
- three expert judge entry points
- synthesis logic
- unified input and output schemas
- architecture and integration documentation
- a local SLM prototype environment using Ollama

## Current Platform Status

The current prototype already supports the following workflow:

Submission Input  
→ Judge 1  
→ Judge 2  
→ Judge 3  
→ Synthesis Layer  
→ Outputs and Logs

At the current stage:
- **Judge 1** is connected to a real local SLM backend through Ollama
- **Judge 2** is currently a mock module
- **Judge 3** is currently a mock module

This means the repository is no longer only a mock pipeline. It already includes an initial local SLM based expert module.

## Expert Judge Roles

### Judge 1: Technical Safety Judge
Focuses on technical safety risks such as prompt injection, jailbreak susceptibility, adversarial robustness, and unsafe behavior.

### Judge 2: Governance and Policy Alignment Judge
Focuses on governance, policy alignment, compliance related concerns, and control documentation.

### Judge 3: Operational and System Risk Judge
Focuses on workflow reliability, deployment context, operational vulnerabilities, and system level concerns.

## Local SLM Environment

The project currently uses **Ollama** as the local SLM runtime.

### Current Local Model
- Runtime: `Ollama`
- Model: `llama3.2:3b`
- API endpoint: `http://localhost:11434/api/generate`

### Why This Exists
The DGX Spark / NYU sandbox target environment is not yet available for direct development, so this repository includes a local substitute SLM environment for development and integration preparation.

## Repository Structure

```text
unicc-ai-safety-lab-sp26/
├── app/
│   ├── judge1.py
│   ├── judge2.py
│   ├── judge3.py
│   ├── main.py
│   ├── orchestrator.py
│   └── synthesis.py
│
├── data/
│   ├── sample_submission.json
│   ├── sample_submission_low.json
│   └── sample_submission_critical.json
│
├── docs/
│   ├── integration_rules.md
│   ├── judge_role_definition.md
│   └── system_architecture.md
│
├── logs/
│   └── pipeline_log.json
│
├── outputs/
│   ├── full_result.json
│   ├── judge1_output.json
│   ├── judge2_output.json
│   ├── judge3_output.json
│   └── synthesis_output.json
│
├── schemas/
│   ├── system_input_schema.md
│   ├── expert_output_schema.md
│   └── synthesis_output_schema.md
│
└── requirements.txt
```
## Requirements

### Install Python dependencies:
```text
pip install -r requirements.txt
```

## Local SLM Setup
###1. Install Ollama

Install Ollama on your local machine.

###2. Pull the model
```text
ollama pull llama3.2:3b
```

### 3. Test the model locally
```text
ollama run llama3.2:3b
```

### 4. Test the API
```text
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Return a short JSON object with status ok",
  "stream": false
}'
```

## How to Run the Project
### Option 1: Run the local pipeline directly

Go to the app folder:
```text
cd app
```
Run the pipeline:
```text
python main.py
```
If needed, use:
```text
python3 main.py
```
### Option 2: Run the upload API

From the project root directory, run:
```text
python3 -m uvicorn app.api:app --reload --port 8000
```
Then open:
```text
http://127.0.0.1:8000/docs
```
Use the POST /submit endpoint to test metadata submission and file upload.
## Input Cases

### The data/ folder contains sample submission cases for testing.

- sample_submission.json
Baseline case
- sample_submission_low.json
Lower risk case
- sample_submission_critical.json
Higher risk or critical case

These are submission cases, not expert judges. Each submission case is evaluated by all three judges.

## Output Files

### After running the pipeline, the system generates:

- outputs/judge1_output.json
- outputs/judge2_output.json
- outputs/judge3_output.json
- outputs/synthesis_output.json
- outputs/full_result.json

And a log file:

- logs/pipeline_log.json

## Integration Rules

The platform is intentionally structured so that future expert module integration happens at the judge layer.

Future module integration should happen in:

- app/judge1.py
- app/judge2.py
- app/judge3.py

The following files should remain stable unless necessary:

- app/main.py
- app/orchestrator.py
- app/synthesis.py

All future expert modules should preserve the common input and output schemas documented in the schemas/ folder.

## Current Handoff Guidance

For the next development stage:

- Project 1 has prepared the platform skeleton, documentation, schemas, local SLM environment, and upload API prototype
- all three judge slots are now connected to the local SLM backend with different expert prompts
- the upload API provides a working local submission path for metadata and file based testing
- future work can continue by refining judge prompts, replacing prototype logic with more specialized expert module logic, and expanding the local environment toward the DGX Spark and sandbox target environment
## Notes

This repository is currently a development stage prototype. It is intended to support:

- architecture definition
- local SLM experimentation
- expert module integration
- upload and submission flow prototyping
- future portability to the NYU DGX Spark and sandbox environment
