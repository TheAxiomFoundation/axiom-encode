"""Helpers for extracting USC statute text and subsection fragments from XML."""

from __future__ import annotations

import html as html_module
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CitationParts:
    """Structured representation of a USC citation."""

    title: str
    section: str
    fragments: tuple[str, ...] = ()


def parse_usc_citation(citation: str) -> CitationParts:
    """Parse a USC citation or slash path into title, section, and fragments."""
    cleaned = citation.strip().replace("§", "")

    if "/" in cleaned and "USC" not in cleaned.upper():
        parts = [part for part in cleaned.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Could not parse citation: {citation}")
        return CitationParts(
            title=parts[0],
            section=parts[1],
            fragments=tuple(parts[2:]),
        )

    cleaned = re.sub(r"\bUSC\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    pieces = cleaned.split(" ", 1)
    if len(pieces) != 2:
        raise ValueError(f"Could not parse citation: {citation}")

    title = pieces[0]
    remainder = pieces[1].replace(" ", "")
    match = re.match(r"^(?P<section>[0-9A-Za-z.-]+)(?P<tail>(?:\([^)]+\))*)$", remainder)
    if not match:
        raise ValueError(f"Could not parse citation: {citation}")

    fragments = tuple(re.findall(r"\(([^)]+)\)", match.group("tail")))
    return CitationParts(title=title, section=match.group("section"), fragments=fragments)


def citation_to_source_path(citation: str | CitationParts) -> str:
    """Convert a citation into the XML subsection source path format."""
    parts = citation if isinstance(citation, CitationParts) else parse_usc_citation(citation)
    path_parts = [parts.title, parts.section, *parts.fragments]
    return "usc/" + "/".join(path_parts)


def citation_to_relative_rac_path(citation: str | CitationParts) -> Path:
    """Convert a citation into the repo-relative RAC output path."""
    parts = citation if isinstance(citation, CitationParts) else parse_usc_citation(citation)
    if not parts.fragments:
        return Path(parts.title) / parts.section / f"{parts.section}.rac"
    return Path(parts.title) / parts.section / Path(*parts.fragments).with_suffix(".rac")


def uscode_xml_path(xml_root: Path, citation: str | CitationParts) -> Path:
    """Resolve the USC XML file for a citation."""
    parts = citation if isinstance(citation, CitationParts) else parse_usc_citation(citation)
    return Path(xml_root) / f"usc{parts.title}.xml"


def extract_section_text_from_xml(xml_path: Path, section: str) -> str | None:
    """Extract the flattened text for an entire section from USC XML."""
    content = xml_path.read_text()
    title = xml_path.stem.replace("usc", "")
    identifier = f"/us/usc/t{title}/s{section}"
    start_pattern = rf'<section[^>]*identifier="{re.escape(identifier)}"[^>]*>'
    start_match = re.search(start_pattern, content)
    if not start_match:
        return None

    start_pos = start_match.start()
    depth = 0
    end_pos = start_pos
    i = start_pos
    while i < len(content):
        if content[i : i + 8] == "<section":
            depth += 1
        elif content[i : i + 10] == "</section>":
            depth -= 1
            if depth == 0:
                end_pos = i + 10
                break
        i += 1

    xml_section = content[start_pos:end_pos]
    text = re.sub(r"<[^>]+>", " ", xml_section)
    text = html_module.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_subsections_from_xml(xml_path: Path, section: str) -> list[dict]:
    """Extract nested subsections from USC XML."""
    content = xml_path.read_text()
    title = xml_path.stem.replace("usc", "")
    identifier = f"/us/usc/t{title}/s{section}"
    start_pattern = rf'<section[^>]*identifier="{re.escape(identifier)}"[^>]*>'
    start_match = re.search(start_pattern, content)

    if not start_match:
        return []

    start_pos = start_match.start()
    depth = 0
    end_pos = start_pos
    i = start_pos

    while i < len(content):
        if content[i : i + 8] == "<section":
            depth += 1
        elif content[i : i + 10] == "</section>":
            depth -= 1
            if depth == 0:
                end_pos = i + 10
                break
        i += 1

    xml_section = content[start_pos:end_pos]

    def clean(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = html_module.unescape(text)
        return " ".join(text.split()).strip()

    def extract_elements_recursive(xml: str, depth: int = 0) -> list[dict]:
        results = []
        tag_order = ["subsection", "paragraph", "subparagraph", "clause", "subclause"]
        if depth >= len(tag_order):
            return results

        tag = tag_order[depth]
        pattern = rf'<{tag}[^>]*identifier="([^"]+)"[^>]*>'

        for match in re.finditer(pattern, xml):
            ident = match.group(1)
            open_tag = f"<{tag}"
            close_tag = f"</{tag}>"
            nested_depth = 1
            j = match.end()
            close_index = len(xml)
            while j < len(xml) and nested_depth > 0:
                if xml[j : j + len(open_tag)] == open_tag:
                    nested_depth += 1
                    j += len(open_tag)
                    continue
                if xml[j : j + len(close_tag)] == close_tag:
                    nested_depth -= 1
                    if nested_depth == 0:
                        close_index = j
                        break
                    j += len(close_tag)
                    continue
                j += 1

            elem_xml = xml[match.end() : close_index]
            heading_match = re.search(
                r"<heading[^>]*>(.*?)</heading>", elem_xml, re.DOTALL
            )
            content_match = re.search(r"<content>(.*?)</content>", elem_xml, re.DOTALL)

            heading = clean(heading_match.group(1)) if heading_match else ""
            body = clean(content_match.group(1)) if content_match else ""

            path_parts = ident.split("/")
            try:
                sec_idx = next(i for i, p in enumerate(path_parts) if p.startswith("s"))
                local_path = "/".join(
                    [path_parts[sec_idx][1:]] + path_parts[sec_idx + 1 :]
                )
            except StopIteration:
                local_path = path_parts[-1]

            results.append(
                {
                    "source_path": f"usc/{title}/{local_path}",
                    "heading": heading,
                    "body": body,
                }
            )
            results.extend(extract_elements_recursive(elem_xml, depth + 1))

        return results

    rules = extract_elements_recursive(xml_section)
    sec_heading = re.search(
        r"<heading[^>]*>(.*?)</heading>", xml_section[:500], re.DOTALL
    )
    sec_content = re.search(r"<chapeau>(.*?)</chapeau>", xml_section, re.DOTALL)
    if sec_heading:
        rules.insert(
            0,
            {
                "source_path": f"usc/{title}/{section}",
                "heading": clean(sec_heading.group(1)),
                "body": clean(sec_content.group(1)) if sec_content else "",
            },
        )
    return rules


def find_citation_text(citation: str, xml_root: Path) -> str | None:
    """Find text for a section or subsection citation from local USC XML."""
    parts = parse_usc_citation(citation)
    xml_path = uscode_xml_path(xml_root, parts)
    if not xml_path.exists():
        return None

    if not parts.fragments:
        return extract_section_text_from_xml(xml_path, parts.section)

    target_source_path = citation_to_source_path(parts)
    for rule in extract_subsections_from_xml(xml_path, parts.section):
        if rule.get("source_path") == target_source_path:
            label = parts.fragments[-1]
            heading = (rule.get("heading") or "").strip()
            body = (rule.get("body") or "").strip()
            components = [f"({label})"]
            if heading:
                components.append(heading)
            if body:
                components.append(body)
            return " ".join(components).strip()

    return None
