# Clinical Mental Health Assistant RAG Chatbot

The Clinical Mental Health Assistant (C-MHA) is an educational MSAI 631-B01
AI/HCI project. It uses retrieval-augmented generation (RAG) to turn a curated,
local knowledge base into concise conversational answers while showing the
local sources used.

> **Important:** C-MHA is an experimental educational prototype, not a medical
> device or a substitute for professional care. It does not diagnose, prescribe,
> or provide emergency support.

## Screenshots

### Local chat interface

![C-MHA local chat interface](docs/images/c-mha-home.png)

### Grounded answer with the retrieved local source

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
  model. No paid inference API is used; a read-only Hugging Face token
  authenticates the model downloads.
- **Empathetic HCI:** Gradio provides a focused chat interface, persistent
  disclaimer, example questions, and visible source labels.
- **Safety before generation:** recognized high-risk first-person self-harm or
  violence disclosures bypass both retrieval and the language model and receive
  a deterministic escalation response.
- **Low-confidence behavior:** the assistant declines when retrieved material
  does not meet the configured relevance threshold.
- **Privacy-conscious defaults:** analytics, saved history, and flagging are
  disabled. Message contents are not intentionally logged.
- **Laptop-friendly:** the default `Qwen2.5-0.5B-Instruct` generator and
  `all-MiniLM-L6-v2` embedding model run on CPU. Models are downloaded on first
  use and then read from the local Hugging Face cache.
- **Transparent ingestion:** PDF page numbers and source filenames are retained;
  a content fingerprint automatically invalidates a stale index.

## Architecture

```text
User message
    |
    v
Deterministic safety router ------> urgent-support response
    |
    v
Sentence Transformer query embedding
    |
    v
FAISS similarity search over reviewed local files
    |
    +------> insufficient relevance: safe decline
    |
    v
LangChain prompt + local Hugging Face generator
    |
    v
Answer + retrieved local source labels
```

The persisted FAISS binary is paired with JSON metadata rather than a Python
pickle, avoiding unsafe pickle deserialization.

## Project structure

```text
C-MHA/
├── data/                    # Reviewed knowledge files and starter corpus
├── src/
│   ├── config.py            # Validated environment configuration
│   ├── documents.py         # PDF/text/Markdown loading and chunking
│   ├── generator.py         # LangChain + local Hugging Face generation
│   ├── index.py             # Safe FAISS persistence and retrieval
│   ├── ingest.py            # Ingestion CLI
│   ├── rag.py               # End-to-end RAG coordination
│   └── safety.py            # Model-independent crisis routing
├── tests/                   # Dependency-light unit tests
├── vector_db/               # Generated index (ignored by Git)
├── app.py                   # Gradio application
└── requirements.txt
```

## Run locally

Python 3.10 or newer is required. Python 3.10-3.12 is recommended for broad ML
package compatibility.

### Windows PowerShell: complete setup

The commands below avoid PowerShell activation-policy problems by calling the
virtual environment's Python executable directly.

#### 1. Install the prerequisites

- [Git for Windows](https://git-scm.com/download/win)
- [Python 3.12](https://www.python.org/downloads/)
- A free [Hugging Face account](https://huggingface.co/join)

Confirm that Git and Python are available:

```powershell
git --version
py -3.12 --version
```

#### 2. Clone the repository and select the feature branch

For a new checkout:

```powershell
Set-Location (Join-Path $env:USERPROFILE "Documents")
git clone https://github.com/msai-631-hci-ai-org/clinical-mental-health-assistant-chatbot-repo.git
Set-Location "clinical-mental-health-assistant-chatbot-repo"
git switch feature/kalirajann
```

For an existing checkout, open PowerShell in the repository and run:

```powershell
git switch feature/kalirajann
```

If Git reports `detected dubious ownership`, trust only this exact repository
path, then retry the branch command:

```powershell
$repo = (Get-Location).Path.Replace("\", "/")
git config --global --add safe.directory $repo
git switch feature/kalirajann
```

Do not configure `safe.directory` as `*`.

#### 3. Create a short-path virtual environment

Keeping the environment outside the repository reduces the chance of Windows
path-length errors:

```powershell
$venv = Join-Path $env:USERPROFILE "cmha-venv"
py -3.12 -m venv $venv
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
& $python -m pip check
```

`pip check` should print `No broken requirements found.`

#### 4. Create a Hugging Face read token

1. Sign in to Hugging Face.
2. Open [Settings > Access Tokens](https://huggingface.co/settings/tokens).
3. Select **Create new token**.
4. Give the token a project-specific name such as `cmha-local`.
5. Choose **Read** access. This application only downloads models and does not
   need write access.
6. Copy the token when it is shown. Treat it like a password.

#### 5. Add the token to `.env`

Create the local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder with the copied token:

```dotenv
HF_TOKEN=hf_your_actual_read_token
```

Save and close Notepad. Do not add quotes or spaces around the token. The
repository ignores `.env`; never commit or share that file, paste its contents
into an issue, or include the token in a screenshot.

Verify that the application can see a token without printing its value:

```powershell
& $python -c "from src.config import Settings; Settings.from_env(); import os; print('HF_TOKEN configured:', bool(os.getenv('HF_TOKEN')))"
```

The expected result is `HF_TOKEN configured: True`.

#### 6. Build the local RAG index

The starter knowledge base is already under `data/`. Build its FAISS index:

```powershell
& $python -m src.ingest --force
```

The first run downloads the pinned `all-MiniLM-L6-v2` embedding model. A
successful starter build reports that 6 chunks were indexed under `vector_db/`.

#### 7. Run the automated checks

```powershell
& $python -m compileall -q app.py src tests
& $python -m unittest discover -s tests -v
```

#### 8. Start the application

```powershell
& $python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in a browser. Keep the
PowerShell window open while using the application. Press `Ctrl+C` in that
window to stop it.

The first submitted question downloads the pinned
`Qwen2.5-0.5B-Instruct` generation model. On a CPU-only computer, the initial
download and first response can take several minutes. Later runs reuse the
local Hugging Face cache.

#### 9. Run it again later

From the repository directory:

```powershell
$venv = Join-Path $env:USERPROFILE "cmha-venv"
$python = Join-Path $venv "Scripts\python.exe"
& $python app.py
```

Re-run `& $python -m src.ingest --force` after adding or editing knowledge files
under `data/`.

### macOS or Linux

```bash
git clone https://github.com/msai-631-hci-ai-org/clinical-mental-health-assistant-chatbot-repo.git
cd clinical-mental-health-assistant-chatbot-repo
git switch feature/kalirajann
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and add a read-only token after HF_TOKEN=.
python -m src.ingest --force
python -m unittest discover -s tests -v
python app.py
```

### Local setup troubleshooting

- **`401 Unauthorized` or `403 Forbidden`:** verify that `.env` contains a
  current read token named exactly `HF_TOKEN`, then stop and restart the app.
- **Token verification prints `False`:** confirm the file is named `.env`, not
  `.env.txt`, and that it is in the repository root beside `app.py`.
- **Windows path-length error:** use the short external environment path shown
  above, and consider cloning the repository closer to the drive root.
- **Port 7860 is already in use:** stop the older app process with `Ctrl+C`
  before launching another instance.
- **Slow first response:** model downloads and CPU generation take time. Watch
  the PowerShell output and allow the initial downloads to finish.
- **Knowledge changes are missing:** rebuild with
  `& $python -m src.ingest --force`, then restart the app.

## Hugging Face Spaces deployment

This repository is structured for a Gradio Hugging Face Space: `app.py` is the
entry point and `requirements.txt` declares the runtime dependencies. Creating
the shared Space, selecting its visibility, and approving its public release
are team-owned deployment steps and are not claimed as completed by this
feature branch.

After the team approves deployment:

1. Create a new Space with the **Gradio** SDK.
2. Connect or upload this repository and select the intended branch.
3. In the Space settings, add `HF_TOKEN` as a **Secret**, using a read-only
   project token. Never place the token in a committed file.
4. Use CPU Basic or another hardware tier with sufficient RAM for the pinned
   models.
5. Confirm the build succeeds, the starter corpus indexes, source labels appear,
   and the deterministic safety prompts pass before sharing the Space URL.
6. Record the approved Space URL and release-review date in this README.

The app does not require a paid inference endpoint; model inference runs in the
Space process. The local setup remains the required validation path before a
cloud release.

## Curate the knowledge base

The included `data/starter_knowledge_base.md` is a small demonstration corpus
paraphrased from current NIMH, CDC, and 988 Lifeline pages. It is not a complete
clinical dataset.

1. Add reviewed `.pdf`, `.md`, or `.txt` files under `data/`.
2. Record the authoritative URL, review date, intended audience, and permission
   to use each source.
3. Run `python -m src.ingest --force`.
4. Validate representative, out-of-scope, adversarial, and crisis prompts with
   qualified human reviewers before any public deployment.

Never treat arbitrary uploads or model-generated text as trusted clinical
source material.

## Configuration

Copy `.env.example` to `.env` to override defaults:

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | none | Read-only Hugging Face token used to download models |
| `CMHA_DATA_DIR` | `data` | Knowledge source directory |
| `CMHA_INDEX_DIR` | `vector_db` | Generated FAISS/JSON files |
| `CMHA_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CMHA_EMBEDDING_REVISION` | pinned commit | Reviewed embedding-model revision |
| `CMHA_GENERATION_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Local instruction model |
| `CMHA_GENERATION_REVISION` | pinned commit | Reviewed generation-model revision |
| `CMHA_DEVICE` | `-1` | CPU (`-1`) or CUDA device number |
| `CMHA_TOP_K` | `3` | Maximum retrieved chunks |
| `CMHA_MIN_RELEVANCE_SCORE` | `0.25` | Minimum cosine similarity |
| `CMHA_CHUNK_SIZE` | `700` | Approximate chunk characters |
| `CMHA_CHUNK_OVERLAP` | `100` | Overlapping characters |
| `CMHA_MAX_HISTORY_MESSAGES` | `4` | Recent messages placed in prompt |

## Test

The core tests deliberately use fakes and the Python standard library, so
safety and RAG control flow can be checked without downloading ML models:

```bash
python -m compileall -q app.py src tests
python -m unittest discover -s tests -v
```

For a release candidate, also install all dependencies, rebuild the index, open
the Gradio interface, and complete human safety/usability evaluation.

## Known limitations and ethical safeguards

- Keyword/phrase safety routing is a backstop, not a validated clinical risk
  classifier. It can miss danger or produce false positives.
- Personal diagnosis and medication-change requests are routed to deterministic
  boundary responses before retrieval or generation.
- Retrieval relevance does not prove clinical correctness or completeness.
- Small local language models can still misstate or omit information.
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
