"""
Embedded agent prompts for Axiom Encode.

All prompts are self-contained -- no dependency on external plugins.
The public prompt surface is RuleSpec-only.
"""

from .encoder import (
    ENCODER_PROMPT,
    SOURCE_SCOPE_PROTOCOL,
    get_encoder_prompt,
)

__all__ = [
    "ENCODER_PROMPT",
    "SOURCE_SCOPE_PROTOCOL",
    "get_encoder_prompt",
]
