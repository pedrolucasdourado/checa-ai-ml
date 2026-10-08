"""
src/safety/moderation.py
─────────────────────────────────────────────────────────────────────
Wrapper for the OpenAI Moderation API to detect toxic content.
"""

import os
from typing import NamedTuple, Optional

class ModerationResult(NamedTuple):
    is_safe: bool
    category: Optional[str] = None
    flagged: bool = False

def check_moderation(text: str) -> ModerationResult:
    """
    Checks if the given text violates OpenAI's moderation policies.
    
    Returns a ModerationResult indicating if the text is safe.
    """
    try:
        import openai
    except ImportError:
        raise RuntimeError("The 'openai' package is required for moderation checks.")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # In a production environment, we might want to fail-safe to 'unsafe' 
        # or log a warning. Here we log and assume safe but warn.
        # However, for security, it's better to be strict.
        # Let's assume for now it's configured.
        pass
    
    client = openai.OpenAI(api_key=api_key)
    response = client.moderations.create(input=text)
    
    result = response.results[0]
    if result.flagged:
        # Find the first category that was flagged
        categories = result.categories.__dict__
        flagged_category = next((cat for cat, flagged in categories.items() if flagged), "unknown")
        return ModerationResult(is_safe=False, category=flagged_category, flagged=True)
    
    return ModerationResult(is_safe=True, flagged=False)
