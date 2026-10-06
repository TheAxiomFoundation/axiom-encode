"""Shared currency vocabulary for numeric extraction and recall cleanup."""

CURRENCY_MARKER_FRAGMENT = (
    r"(?:[$£€¥₹]|"
    r"(?:euros?|eur|dollars?|usd|pounds?|gbp|cad|aud|chf)\b|"
    r"(?:(?:u\.?\s*s\.?|united\s+states|canadian|australian)\s+dollars?|"
    r"swiss\s+francs?)\b)"
)
