"""
src/safety/validator.py
─────────────────────────────────────────────────────────────────────
Validates the generated output for safety and quality.
"""

from src.safety.moderation import check_moderation, ModerationResult

def validate_output(output_text: str) -> ModerationResult:
    """
    Validates that the generated counter-narrative is safe.
    Currently leverages the Moderation API.
    """
    # Output validation might have different thresholds or additional checks
    # like checking for banned words or ensuring it's not too short/long.
    return check_moderation(output_text)
