"""Shared resource bounds for validator feedback carried into model retries."""

from axiom_encode.checks import retype

VALIDATION_RETRY_FEEDBACK_MAX_ITEMS = 12
VALIDATION_RETRY_FEEDBACK_MAX_ITEM_CHARS = 16_000
VALIDATION_RETRY_FEEDBACK_MAX_TOTAL_CHARS = 64_000
VALIDATION_RETRY_FEEDBACK_ELISION = "\n...[diagnostic middle elided]...\n"
VALIDATION_RETRY_CHECK_FIX_MAX_CHARS = 8_000


def bounded_validation_retry_feedback_item(raw_item: str) -> str:
    """Retain both ends of one validator diagnostic within its prompt budget."""

    item = raw_item.strip()
    limit = VALIDATION_RETRY_FEEDBACK_MAX_ITEM_CHARS
    if len(item) <= limit:
        return retype(raw_item, item)
    available = limit - len(VALIDATION_RETRY_FEEDBACK_ELISION)
    head_chars = available // 2
    tail_chars = available - head_chars
    return retype(
        raw_item,
        item[:head_chars] + VALIDATION_RETRY_FEEDBACK_ELISION + item[-tail_chars:],
    )
