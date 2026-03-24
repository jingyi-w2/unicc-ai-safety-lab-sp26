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

Install Python dependencies:

~~~bash
pip install -r requirements.txt
```

future portability to the NYU DGX Spark / sandbox environment
