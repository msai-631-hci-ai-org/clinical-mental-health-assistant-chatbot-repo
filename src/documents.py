import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "./data"          # Directory containing mental health PDF documents
FAISS_SAVE_PATH = "./vector_db" # Directory to persist the vector store

def create_vector_db():
    # Create directories if they don't exist
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(FAISS_SAVE_PATH, exist_ok=True)

    print("1. Loading documents from data directory...")

    # --- Start Debugging Additions ---
    print(f"Current working directory: {os.getcwd()}")
    abs_data_path = os.path.abspath(DATA_PATH)
    print(f"Resolved DATA_PATH: {abs_data_path}")
    if not os.path.exists(abs_data_path):
        print(f"Error: The data directory '{abs_data_path}' does not exist.")
        # return # Cannot proceed if data directory itself is missing, but we want to show URL loading if data dir is empty.
    else:
        print(f"Data directory '{abs_data_path}' exists.")
        try:
            files_in_data = os.listdir(abs_data_path)
            if not files_in_data:
                print(f"Warning: Data directory '{abs_data_path}' is empty.")
            else:
                print(f"Files found in '{abs_data_path}': {files_in_data}")
        except Exception as e:
            print(f"Error listing contents of '{abs_data_path}': {e}")
            # return # Cannot proceed if data directory itself is missing, but we want to show URL loading if data dir is empty.
    # --- End Debugging Additions ---

    documents = []

    # Load PDFs
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    # Load Text files
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())

    # Load Markdown files
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=UnstructuredMarkdownLoader)
    documents.extend(md_loader.load())

    print(f"Loaded {len(documents)} document pages/files (PDFs, TXT, MD) from local directory.")

    # --- Add URL Loading ---
    print("Loading documents from URLs...")
    # Replace with desired URLs
    urls = [
        "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
        "https://www.nimh.nih.gov/health/topics/depression",
        "https://www.un.org/en/global-issues/mental-health",
        "https://www.nimh.nih.gov/health/find-help",
        "https://www.apa.org/ptsd-guideline/patients-and-families/cognitive-behavioral",
        "https://www.cdc.gov/mental-health/living-with/index.html",
    ]
    web_loader = WebBaseLoader(urls)
    web_documents = web_loader.load()
    documents.extend(web_documents)
    print(f"Loaded {len(web_documents)} document pages from URLs.")
    # -- End URL Loading ---

    print(f"Total loaded documents: {len(documents)}.")

    print("2. Splitting text into clinical chunks...")
    # Recursive splitting preserves paragraph context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("3. Generating embeddings and building FAISS index...")
    # Free, local, CPU-friendly embedding model (384 dimensions)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create vector database and persist to disk
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_SAVE_PATH)
    print(f"Success! FAISS index saved locally at '{FAISS_SAVE_PATH}'.")