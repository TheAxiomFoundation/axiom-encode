"""Conservative, offset-preserving printed form metadata recognition.

A complete title/code/revision agreement is required. These spans are display
coordinates, never evidence that any substantive source condition is satisfied.
"""

import re

_TITLE = re.compile(
    r"\A(?P<publisher>[A-Z][A-Za-z ]{2,60}) Schedule (?P<schedule>[A-Z]) "
    r"(?P<year>(?:19|20)\d{2})(?P<dash> [–—-] )"
    r"[A-Z][A-Za-z ,'-]{3,120}\n"
)
_CODE = re.compile(
    r"(?m)^(?:[A-Z][ ]*){2,8}-[ ]*\d{1,4}\."
    r"(?:[ ]*[A-Z]\.){1,4}[ ]*(?P<schedule>[A-Z])[ ]*-[ ]*[A-Z] "
    r"\((?P<year>(?:19|20)\d{2})-(?:0[1-9]|1[0-2])\)$"
)
_REFERENCES = re.compile(
    r"\blines? +(?P<labels>[1-9]\d{0,2}"
    r"(?:(?:, *| +(?:and|or) +)[1-9]\d{0,2})*)(?![\d%]|\.\d)",
    re.I,
)


def form_display_spans(text: str) -> tuple[tuple[int, int], ...]:
    title = _TITLE.match(text)
    if title is None:
        return ()
    codes = list(_CODE.finditer(text, title.end(), min(len(text), title.end() + 160)))
    if len(codes) != 1:
        return ()
    code = codes[0]
    if any(title[k] != code[k] for k in ("schedule", "year")):
        return ()
    # Only whitespace may separate the title and its complete form identifier.
    if text[title.end() : code.start()].strip():
        return ()
    spans = [title.span("year"), title.span("dash"), code.span()]
    tail = re.search(
        r"(?m)(?<=\n\n)(?P<labels>(?:[1-9]\d{0,2}\n){3,20})\s*"
        + re.escape(title["schedule"])
        + r"\s*\Z",
        text,
    )
    if tail:
        labels = [int(x) for x in tail["labels"].split()]
        references = {
            int(n)
            for match in _REFERENCES.finditer(text[: tail.start()])
            for n in re.findall(r"\d+", match["labels"])
        }
        if labels == sorted(set(labels)) and set(labels) <= references:
            spans.append(tail.span("labels"))
    return tuple(spans)


def mask_form_display(text: str) -> str:
    """Blank only corroborated metadata; retain every newline and source offset."""
    chars = list(text)
    for start, end in form_display_spans(text):
        for i in range(start, end):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)
