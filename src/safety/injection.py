"""
src/safety/injection.py
─────────────────────────────────────────────────────────────────────
Heuristic-based prompt injection and jailbreak detection.
"""

import re
from typing import NamedTuple, Optional

class InjectionResult(NamedTuple):
    is_safe: bool
    pattern: Optional[str] = None

# Common jailbreak/injection patterns (English and Portuguese)
INJECTION_PATTERNS = [
    # English
    r"ignore all previous instructions",
    r"ignore the above",
    r"system prompt",
    r"act as a",
    r"you are now a",
    r"disregard previous",
    r"override previous",
    r"tell me how to build a bomb", # Explicit harmful intent
    r"bypass safety",
    r"jailbreak",
    # Portuguese
    r"ignore todas as instruções",
    r"ignore as instruções anteriores",
    r"ignore o prompt",
    r"desconsidere as instruções",
    r"esqueça as instruções",
    r"aja como um",
    r"você agora é um",
    r"finja que é um",
    r"me diga como construir uma bomba",
    r"como hackear",
    r"burle a segurança",
    r"ignorar instruções",
]

def check_injection(text: str) -> InjectionResult:
    """
    Analyzes text for common prompt injection patterns.
    """
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return InjectionResult(is_safe=False, pattern=pattern)
    
    return InjectionResult(is_safe=True)
