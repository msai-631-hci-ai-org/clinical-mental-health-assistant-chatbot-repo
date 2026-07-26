import re

# Emergency keywords triggering instant fallback
CRISIS_KEYWORDS = [
    r"\bsuicide\b", r"\bsuicidal\b", r"\bkill myself\b", r"\bend my life\b",
    r"\bself-harm\b", r"\bhurt myself\b", r"\boverdose\b", r"\bwant to die\b",
    r"\bcut myself\b", r"\bcut me\b", r"\bcutting\b", r"\bcutting myself\b",
    r"\b(i'm|im|i am) going to cut\b", r"\b(i'm|im|i am) going to hurt myself\b",
    r"\b(i'm|im|i am) going to kill myself\b",
    r"\bfeeling hopeless\b", r"\bno reason to live\b", r"\bdon't want to be here\b",
    r"\bget rid of me\b", r"\bdisappear\b", r"\bnot wake up\b",
    r"\bending it all\b", r"\bcan't go on\b", r"\bcan't take it anymore\b",
    r"\btired of living\b", r"\bdead\b"
]

CRISIS_RESPONSE = (
    "⚠️ **CRISIS ALERT & IMMEDIATE SUPPORT NEEDED**\n\n"
    "If you or someone you know is in immediate danger or experiencing a mental health crisis, "
    "please reach out for support right away:\n\n"
    "- **Call or Text 988** to reach the Suicide & Crisis Lifeline (US & Canada, Available 24/7, Free & Confidential).\n"
    "- **Text HOME to 741741** to connect with the Crisis Text Line.\n"
    "- **International Crisis Resources:** [https://findahelpline.com/](https://findahelpline.com/)\n"
    "- **Call 911** or go to the nearest emergency room.\n\n"
    "*This AI Assistant is not a healthcare provider and cannot handle emergencies.*"
)

def check_crisis_intent(user_input: str) -> bool:
    """Returns True if user input contains self-harm or acute crisis indicators."""
    text_lower = user_input.lower()
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, text_lower):
            return True
    return False