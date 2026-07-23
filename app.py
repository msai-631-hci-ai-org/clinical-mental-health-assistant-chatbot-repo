"""Gradio entry point for the Clinical Mental Health Assistant."""

from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Hub environment variables must be loaded before Gradio or Transformers can
# import huggingface_hub and initialize its authentication configuration.
load_dotenv(Path(__file__).with_name(".env"), override=False)

# Disable optional framework/model telemetry before importing those libraries.
# Model downloads still require access to the Hugging Face Hub on first use.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import gradio as gr

from src.config import Settings
from src.rag import RAGAssistant

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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
may be appropriate. Answers are limited to the local, curated knowledge base and
show which local sources were retrieved.
"""


@lru_cache(maxsize=1)
def get_assistant() -> RAGAssistant:
    """Create one lazy assistant instance for the process."""

    return RAGAssistant(Settings.from_env())


def respond(message: str, history: list[dict[str, Any]]) -> str:
    """Return a grounded answer without logging message contents."""

    if not message or not message.strip():
        return "Please enter a question."

    try:
        return get_assistant().answer(message, history).to_markdown()
    except Exception:
        # Avoid exposing stack traces, environment details, or user text in UI.
        logger.exception("The assistant could not complete a request")
        return (
            "I could not complete that request. Please confirm that the "
            "dependencies are installed and run `python -m src.ingest` to "
            "build the local knowledge index."
        )


def build_demo() -> gr.Blocks:
    """Build the privacy-conscious Gradio interface."""

    with (
        gr.Blocks(
            title="C-MHA | Clinical Mental Health Assistant",
            analytics_enabled=False,
        ) as demo,
        gr.Column(),
    ):
        gr.Markdown("# Clinical Mental Health Assistant (C-MHA)")
        gr.Markdown(DISCLAIMER)
        gr.Markdown(WELCOME)

        chatbot_options: dict[str, Any] = {
            "height": 480,
            "placeholder": "Your conversation stays in this running process.",
        }
        chatbot_parameters = inspect.signature(gr.Chatbot).parameters
        if "type" in chatbot_parameters:
            # Gradio 5 supports messages explicitly; Gradio 6 makes them
            # the only/default format and removes this argument.
            chatbot_options["type"] = "messages"
        if "show_copy_button" in chatbot_parameters:
            chatbot_options["show_copy_button"] = True
        elif "buttons" in chatbot_parameters:
            chatbot_options["buttons"] = ["copy", "copy_all"]
        if "feedback_options" in chatbot_parameters:
            chatbot_options["feedback_options"] = []

        chatbot = gr.Chatbot(**chatbot_options)
        interface_options: dict[str, Any] = {
            "fn": respond,
            "chatbot": chatbot,
            "examples": [
                "What is the difference between ordinary worry and an anxiety disorder?",
                "What are some small ways to support my mental health?",
                "When should someone consider talking with a professional?",
            ],
            "cache_examples": False,
            "save_history": False,
            "flagging_mode": "never",
            "autofocus": True,
        }
        if "type" in inspect.signature(gr.ChatInterface).parameters:
            interface_options["type"] = "messages"
        gr.ChatInterface(
            **interface_options,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
