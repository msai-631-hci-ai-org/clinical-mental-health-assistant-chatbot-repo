# Clinical Mental Health Assistant RAG Chatbot

The Clinical Mental Health Assistant (C-MHA) is an educational MSAI 631-B01
AI/HCI project. It uses retrieval-augmented generation (RAG) to turn a curated
knowledge base into a concise conversational agent that answers users questions while showing clinical sources used.

> **Important:** C-MHA is an experimental educational prototype, not a medical
> device or a substitute for professional care. It does not diagnose, prescribe,
> or provide emergency support.

## Screenshots

### Chat interface

![C-MHA chat interface](docs/images/c-mha-home.png)

### Grounded answer with the retrieved source

![C-MHA grounded RAG answer](docs/images/c-mha-rag-answer.png)

## Team

- Derrick Kyei
- Kalirajan Natarajan
- Obinna Asoluka
- Stacey Scott

## Implemented requirements

- **Grounded responses:** normalized Sentence Transformer embeddings and a local
  FAISS index retrieve only from reviewed files in `data/`.
- **Local generation:** LangChain orchestrates a locally downloaded Hugging Face
  model need be if there is no key or token usage. There is no paid inference API is used in this app.
- **Empathetic HCI:** Gradio provides a focused chat interface, persistent
  disclaimer, example questions, and visible source labels.
- **Safety before generation:** recognized high-risk first-person self-harm or
  violence disclosures bypass both retrieval and the language model and receive
  a deterministic escalation response.
- **Low-confidence behavior:** the assistant declines when retrieved material
  does not meet the configured relevance threshold.
- **Privacy-conscious defaults:** analytics, saved history, and flagging are
  disabled. Message contents are not intentionally logged.
- **LLM-LPU:** Utilizes Groq LLM LPU for fast responses. Fallbacks to Hugging Face Local LLM if API KEY isn't in environment.
- **Transparent ingestion:** PDF page numbers and source filenames are retained;
  a content fingerprint automatically invalidates a stale index.

## Architecture

### 1. Build Time / Setup Phase (Runs Once)

* **File:** `documents.py`
* **When it runs:** **Manually executed ONCE** before launching or deploying the app (or executed locally before pushing to Hugging Face).
* **What it does:** 
1. Reads all clinical .pdf, .md, or .txt documents from the `./data` folder.
2. Generates vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
3. Saves the persistent vector database to disk inside a folder named `./vector_db`.
* **Deployment Note:** Once `documents.py` runs, it produces the `./vector_db` folder (containing files like `index.faiss` and `index.pkl`). Commit and push `./vector_db` folder to Hugging Face Spaces alongside Python scripts. `documents.py` is meant to be ran once.  


### 2. Runtime / Execution Phase (Runs Continuously)

* **Files:** `app.py`, `rag.py`, and `safety.py`
* **When they run:** `app.py` is Automatically started by Hugging Face Spaces (or for local testing run `app.py` as well) when the Space boots up, and continuously executed whenever a user sends a prompt.

#### **How the Runtime Files Interact:**

```
 Hugging Face Space Starts / User Accesses Web Interface
                           │
                           ▼
                       [app.py]
         Imports modules and builds Gradio UI
                           │
  User types: "What are symptoms of anxiety?" & clicks Send
                           │
                           ▼
                  [rag.py]
  Receives query and initiates response generation
                           │
                           ├───► Step 1: Calls [safety.py]
                           │      Checks if input contains crisis keywords.
                           │      (If YES: immediately returns crisis support text).
                           │
                           └───► Step 2: Executes RAG Retrieval
                                  Loads persistent ./vector_db index from disk.
                                  Retrieves top k=3 relevant context chunks.
                                  Passes query + chunks to Groq LLM API or Local LLM (much slower).
                                  Returns grounded response back to Gradio UI.

```

### Detailed File Roles at Runtime

| File Name | Role in Hugging Face Space | How it is invoked |
| --- | --- | --- |
| **`app.py`** | **Entry Point & UI** | Hugging Face automatically executes `python app.py` on startup to serve the Gradio web interface to users. |
| **`rag.py`** | **Backend Logic Engine** | `app.py` imports `generate_mental_health_response()` from this file. It initializes the loaded vector store and the Groq LLM connection on app startup or local LLM. |
| **`safety.py`** | **Guardrail Module** | `rag.py` imports `check_crisis_intent()` from this file to evaluate every incoming prompt before any LLM call or context search occurs. |

---

### Summary of Deployment Steps

1. **Locally:** Put .pdf, .md, or .txt in `./data`, run `python documents.py` creates `./vector_db`.
2. **Push to Hugging Face Space:** Push `app.py`, `rag.py`, `safety.py`, `requirements.txt`, and the generated `./vector_db` folder to your Hugging Face Space repository.
3. **Set Environment Variable:** In Hugging Face Space add `GROQ_API_KEY`.
4. **Launch:** Hugging Face automatically runs `app.py`, making the full RAG chatbot live.

The persisted FAISS binary is paired with JSON metadata rather than a Python
pickle, avoiding unsafe pickle deserialization.

## Project structure

```bash
C-MHA/
├── data/                    # Reviewed knowledge files and starter corpus
├── src/
│   ├── documents.py         # PDF/text/Markdown loading and chunking
│   ├── rag.py               # End-to-end RAG coordination
│   └── safety.py            # Model-independent crisis routing
├── vector_db/               # Generated index
├── app.py                   # Gradio application
└── requirements.txt
```

## Run locally

Python 3.10 or newer is required. Python 3.10-3.12 is recommended for broad ML
package compatibility.

### Command Line Setup: complete setup

#### 1. Install the prerequisites

- [Git download](https://git-scm.com/download) (MAC, WINDOWS, LINUX)
- [Git Bash](https://gitforwindows.org/) (If using windows specifically)
- [Python 3.10+](https://www.python.org/downloads/)
- A free [Hugging Face account](https://huggingface.co/join)
- A free [Groq account](https://groq.com)

Confirm that Git and Python are available:

```bash
git --version
python --version
```

#### 2. Clone the repository
```bash
git clone https://github.com/msai-631-hci-ai-org/clinical-mental-health-assistant-chatbot-repo.git

git checkout feature/derrick-enhancements
```

#### 3. Create a GROQ account and API KEY

1. Sign in to GROQ.
2. Open [Settings > Keys](https://console.groq.com/keys).
3. Select **Create API KEY**.
4. Give the Key a project-specific name such as `cmha`.
5. Copy the Key when it is shown. Treat it like a password.
6. If Key is missed, regenerate and keep for use.

#### 4. For local testing, add the API KEY to `.env`

In project workspace directory, create the local environment file:

```bash
touch .env
```

Replace the placeholder with the copied key:

```bash
# Choose
GROQ_API_KEY=your_actual_API_KEY
or
HF_TOKEN=your_actual_TOKEN
# MUST include
HF_DEPLOYMENT=false #set to true for deployment #false for local testing
```

#### 5. Build the RAG index

The starter knowledge base is already under `data/`. Build its FAISS index:

```bash
python documents.py
```

The first run downloads all .pdf, .md, and .txt files along side urls in code for injesting the model. Produces the `./vector_db` folder (containing files like `index.faiss` and `index.pkl`)

#### 6. Start the application

```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in a browser. Keep the
Terminal window open while using the application. Press `Ctrl+C` to stop it.

### Local setup troubleshooting

- **`401 Unauthorized` or `403 Forbidden`:** verify that `.env` contains a
  current read token named exactly `GROQ_API_KEY`, then stop and restart the app.
- **KEY verification prints `False`:** confirm the file is named `.env`, not
  `.env.txt` or `.env.example`, and that it is in the repository root beside `app.py`.
- **`Port 7860` is already in use:** stop the older app process with `Ctrl+C`
  before launching another instance.
- **Cannot access gated repo:** an open-source model may be free to use, but in some cases, seeing an error like this is considered a "gated" model on Hugging Face. This means user will need to explicitly accept its terms of use (usually a license agreement) on the Hugging Face website and generate a `HF_TOKEN=your_actual_token` before download of the local LLM will occur before usage. `transformers` library will be able to authenticate and download the model correctly.

## Hugging Face Spaces deployment

This repository is structured for a Gradio Hugging Face Space: `app.py` is the
entry point and `requirements.txt` declares the runtime dependencies. Creating
the shared Space, selecting its visibility, and approving its public release
are team-owned deployment steps and are not claimed as completed by this
feature branch.

After the team collaborates changes, deployment steps to Hugging Face Spaces are as followed:

1. Create a new Space with the **Gradio** SDK.
2. Connect or upload this repository and select the intended branch.
3. In the Space settings, add `GROQ_API_KEY` or ,`HF_TOKEN`, if local LLM requires, as a **Secret**, using a read-only project token. Never place the token in a committed file.
4. Hugging Face Spaces Gradio defaults to `ZeroGPU`.
5. Confirm the build succeeds, the starter corpus indexes, source labels appear,
   and the deterministic safety prompts pass before sharing the Space URL.
6. Record approved Space URL and release-review date in this README.

## Known limitations and ethical safeguards

- Keyword/phrase safety routing is a backstop, not a validated clinical risk
  classifier. It can miss danger or produce false positives.
- Personal diagnosis and medication-change requests are routed to deterministic
  boundary responses before retrieval or generation.
- Retrieval relevance does not prove clinical correctness or completeness.
- The application does not establish the user's location; U.S. 988 information
  is labeled as U.S.-specific and local emergency/crisis services are advised
  elsewhere.
- No user authentication, encryption-at-rest layer, audit system, or regulatory
  compliance claim is included.
- A clinical, legal, privacy, accessibility, red-team, and human-factors review
  is required before real-world use.

## Source pages used for the starter corpus

- [NIMH: Caring for Your Mental Health](https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health)
- [NIMH: Anxiety Disorders](https://www.nimh.nih.gov/health/topics/anxiety-disorders)
- [NIMH: Depression](https://www.nimh.nih.gov/health/publications/depression)
- [CDC: Managing Stress](https://www.cdc.gov/mental-health/living-with/index.html)
- [988 Lifeline: What to Expect](https://988lifeline.org/get-help/what-to-expect/)

## Disclaimer

C-MHA is an experimental educational prototype. In immediate danger, contact
local emergency services or go to the nearest emergency department. In the
U.S. and its territories, call or text 988 or visit
[988lifeline.org](https://988lifeline.org/).
