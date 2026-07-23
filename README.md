# Clinical Mental Health Assistant RAG Chatbot

This repository is dedicated to MSAI 631-B01 AI/HCI course. The **Clinical Mental Health Assistant (C-MHA)** is a conversational agent developed for interaction and as it utilizes **Retrieval-Augmented Generation (RAG)** framework to provide safe, accessible, and grounded mental health information by synthesizing curated clinical data.

## Table of Contents
- [Team Members](#team-members)
- [Project Overview](#project-overview)
- [The Problem](#the-problem)
- [Key Features](#key-features)
- [Technical Infrastructure/Stack](#technical-infrastructurestack)
- [Project Hierarchy](#project-hierarchy)
- [Ethical Safeguards](#ethical-safeguards)
- [Compliance & Constraints](#compliance--constraints)
- [Disclaimer](#disclaimer)

## Team Members
- Derrick Kyei
- Kalirajan Natarajan
- Obinna Asoluka
- Stacey Scott

## Project Overview
The Clinical Mental Health Assistant Chatbot represents a "shift in agency" in Human-Computer Interaction (HCI). Rather than requiring users to search through complex, static databases like the WHO or NIMH archives, this assistant synthesizes relevant information into an empathetic conversational interaction.The model retrieves data from a trusted vector store before generating a response to the user, the system ensures all answers are **clinically anchored** and minimizes the risk of AI **hallucinations**.

## The Problem
Global mental health is in crisis, with roughly **one in eight people**—over one billion individuals—live with a mental health condition. Despite this, median government spending on mental health is just **two percent** of health budgets, leading to significant treatment gaps. Standalone AI models often provide unreliable or fabricated medical advice; the this chatbot addresses this by grounding its "brain" in verified public-domain resources.

## Key Features
- **Grounded Responses:** Responses are strictly conditioned on retrieved clinical manuals and FAQs
- **Empathetic HCI:** A user interface designed via **Gradio** to provide a supportive and stigma-free environment
- **Hardware Optimized:** The system can execute locally on a typical laptop (8GB-16GB RAM) and is hosted in cloud environment
- **Zero-Token Expenditure:** Built entirely using free-tier resources and open-source models

## Technical Infrastructure/Stack
- **Python:** 3.10+
- **Development Evironment:** Google Colaboratory (Colab)
- **Hosting Platform:** Hugging Face Spaces
- **Orchestration:** LangChain
- **Embeddings:** Sentence-Transformers
- **Vector Database:** ChromaDB or FAISS
- **Candidate Models:** Llama-3 (8B), Mistral-7B, Gemini (Free API), or Llama-3.3 (70B).

## Project Hierarchy
*Note: High-level overview structure of project.*
```bash
C-MHA/
├── data/                   # Knowledge base (WHO/NIMH/CDC PDFs)
├── vector_db/              # Indexed clinical data (ChromaDB/FAISS)
├── src/                    # Backend logic (Ingestion & RAG Pipeline)
├── app.py                  # Gradio UI & Main Application Entry
├── requirements.txt        # Required software libraries
└── README.md               # Project documentation
```

## Ethical Safeguards
To protect vulnerable users, the C-MHA implements the following:
- **Persistent Disclaimers:** The UI features a clear statement that the assistant is not a substitute for professional care.
- **Programmatic Steering:** Prompts are engineered to prioritize clinical referrals (e.g., 988 Lifeline) over diagnostic assertions.
- **Privacy-Preserving Design:** The architecture avoids paid third-party proprietary APIs to ensure user interaction data remains secure.

## Compliance & Constraints
This project adheres to the pedagogical goals outlined for the Group Project:
- Avoids proprietary dependencies (No paid OpenAI tokens).
- Utilizes **Git/GitHub** for transparent, instructor-accessible version control.
- Developed within the free tiers of Hugging Face and Google Colab.

## Disclaimer
**C-MHA is an experimental prototype for educational purposes only.** If you or someone you know is in crisis, please call or text **988** (the U.S. Suicide & Crisis Lifeline) for free, confidential, 24/7 support.