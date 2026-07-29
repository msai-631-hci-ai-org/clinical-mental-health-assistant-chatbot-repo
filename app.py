import gradio as gr
from src.rag import generate_mental_health_response

DISCLAIMER = """
> **Educational prototype — not medical care.** C-MHA provides general,
> source-grounded information. It cannot diagnose, prescribe, or replace a
> licensed professional. Do not include names, contact details, or other
> sensitive information.
>
> **In immediate danger?** Contact local emergency services or go to the
> nearest emergency department. In the U.S. and its territories, call or text
> **988** or visit [988lifeline.org](https://988lifeline.org/).
"""

WELCOME = """
Ask about general mental-health concepts, self-care, or when professional help
may be appropriate. Answers are curated knowledge based and
show which sources were consulted.
"""

def chat_function(message, history):
    # Pass user prompt to RAG engine pipeline
    response = generate_mental_health_response(message)
    return response

# Build Gradio Interface
with gr.Blocks(title="Clinical Mental Health Assistant", theme=gr.Theme.from_hub("VikramSingh178/Webui-Theme")) as demo:
    gr.Markdown("# Clinical Mental Health Assistant (C-MHA)")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(WELCOME)

    gr.ChatInterface(
        fn=chat_function,
        examples=[
            "What is the difference between ordinary worry and an anxiety disorder?",
            "What are some small ways to support my mental health?",
            "When should someone consider talking with a professional?"
        ],
        textbox=gr.Textbox(placeholder="Ask a question about mental health information...", container=False, scale=7),
        chatbot=gr.Chatbot(placeholder="Hello! I'm C-MHA. How can I help you today?"),
        cache_examples=False,
        flagging_mode="never",
        autofocus=True,
        save_history=False
    )

if __name__ == "__main__":
    # Runs locally or serves inside Hugging Face Spaces
    demo.launch()