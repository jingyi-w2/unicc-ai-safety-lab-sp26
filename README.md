# UNICC AI Safety Lab (Spring 2026)

A council based AI Safety Lab for evaluating AI agents before deployment through three independent expert judges and one synthesis layer.

## Overview

This project aims to build a multi module AI Safety Lab for pre deployment evaluation of AI agents in an institutionally controlled environment. The system is designed to assess AI agents from multiple independent perspectives, compare expert outputs, surface disagreements, and generate transparent, auditable recommendations for human review.

The project builds on the top three Fall 2025 capstone solutions, but extends them into a new Spring 2026 architecture based on:
- three independent expert judges
- one critique, arbitration, and synthesis layer
- structured schemas for integration
- portability to the NYU SPS sandbox / DGX Spark environment

## Project Goal

The goal of this project is to develop a council based AI Safety Lab that can:
- evaluate AI agents before deployment
- generate structured safety and governance assessments
- support transparent comparison across multiple expert modules
- provide final recommendations through synthesis logic
- preserve logs, evidence, and audit trails for institutional review

## System Architecture

The proposed system is organized into five layers:

### 1. Input Layer
- agent submission
- metadata form
- evidence upload
- framework selection
- test configuration

### 2. Platform Layer
- SLM runtime environment
- request manager / orchestrator
- schema / interface registry
- artifact store
- log and audit store
- access control

### 3. Expert Layer
- Judge 1
- Judge 2
- Judge 3

### 4. Coordination Layer
- critique engine
- disagreement detection
- arbitration logic
- synthesis engine

### 5. Presentation Layer
- dashboard
- per module results
- executive summary report
- technical detail report
- audit trail

## Repository Structure

```text
unicc-ai-safety-lab-sp26/
├── README.md
├── docs/
├── schemas/
├── assets/
├── sample_data/
├── src/
└── requirements.txt
