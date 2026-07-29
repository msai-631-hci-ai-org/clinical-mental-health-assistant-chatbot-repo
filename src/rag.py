import os
HF_DEPLOYMENT = os.getenv("HF_DEPLOYMENT", "false").lower() == "true"

if HF_DEPLOYMENT:
    import spaces
else:
    class FakeSpaces:
        def GPU(self, fn):
            # No‑op decorator for local testing
            return fn
    spaces = FakeSpaces()
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from src.safety import check_crisis_intent, CRISIS_RESPONSE
from src.documents import create_vector_db
from dotenv import load_dotenv
load_dotenv() #load environment variables to use implicitly

FAISS_SAVE_PATH = "./vector_db"
qa_chain = None #global declaration

def initialize_rag_pipeline():
    # Checks vector_db existence and files/urls for injestion
    create_vector_db()
    
    llm = None
    # Retrieve Groq API Key 
    if 'GROQ_API_KEY' in os.environ:
        # Initialize the Groq LLM
        llm = ChatGroq(
            api_key=os.environ['GROQ_API_KEY'],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_retries=2
            )
    else: 
        print("GROQ_API_KEY not found. Defaulting to local HuggingFacePipeline LLM. May require a HF_TOKEN depending on type of gated model")
        print("WARNING: Local LLMs can be significantly slower and more resource-intensive (especially without GPU) compared to Groq.")

        model_id = "Qwen/Qwen2.5-0.5B-Instruct" # Working model (doesn't require HF_TOKEN) for demonstration

        # Configure 4-bit quantization for efficient memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for better precision if supported
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" # Automatically map model layers to GPU if available, else CPU
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512, # Limit generated output length
            max_length=None,
            temperature=0.2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            return_full_text=False # Only return the generated text, not the input prompt
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"Successfully loaded local LLM: {model_id}")

    if llm is None:
        # This case should ideally not be reached if one of the paths succeeds
        raise ValueError("Could not initialize any LLM. Please check your GROQ_API_KEY setup or local model availability.")

    # 1. Load Local Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Load FAISS Index
    vectorstore = FAISS.load_local(
        FAISS_SAVE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # The 'input' variable will be for the user's question, 'context' for retrieved documents.
    prompt_template_str = """
    You are an educational Clinical Mental Health Assistant.
    Answer the user's question using ONLY the provided clinical context below.
    If the context does not contain enough information to answer, state clearly:
    "I cannot find this information in my verified clinical documents."
    Do NOT diagnose, prescribe, or offer formal therapeutic advice.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    # Use ChatPromptTemplate, as it's commonly used with the new chain factories
    PROMPT = ChatPromptTemplate.from_template(prompt_template_str)

    # Build RAG Chain using create_stuff_documents_chain and create_retrieval_chain
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain

# Global Chain Instance cached for further use
qa_chain = initialize_rag_pipeline()
@spaces.GPU
def generate_mental_health_response(user_query: str) -> str:
    global qa_chain
    # Level 3 Safety Check
    if check_crisis_intent(user_query):
        return CRISIS_RESPONSE

    try:
        response = qa_chain.invoke({"input": user_query})
        answer = response["answer"] # Output key is typically 'answer' now

        # Documents/citations/sources are appened for transparency
        sources = set([doc.metadata.get("source", "NIMH Guidelines") for doc in response["context"]])
        source_text = "\n\n---\n*Sources Consulted:* " + ", ".join(sources)

        return answer + source_text
    except Exception as e:
        return f"An error occurred while processing your query: {str(e)}"