#!/usr/bin/env python3
"""Extract a USC section from local USLM XML files.

Usage:
    python extract_usc_section.py 26 25B
    python extract_usc_section.py 7 2017

Returns plain text suitable for RuleSpec encoding.
"""

import html
import re
import sys
from pathlib import Path

# Default path to USC XML files
USC_XML_PATH = Path.home() / "TheAxiomFoundation" / "axiom" / "data" / "uscode"


def extract_section(title: int, section: str) -> str:
    """Extract a section from USC XML and convert to readable text."""
    xml_file = USC_XML_PATH / f"usc{title}.xml"

    if not xml_file.exists():
        return f"Error: USC Title {title} XML not found at {xml_file}"

    content = xml_file.read_text()

    # Build identifier pattern for section
    identifier = f"/us/usc/t{title}/s{section}"

    # Find the section - look for identifier attribute
    # Pattern: <section ... identifier="/us/usc/t26/s25B">...</section>
    # Use a more careful approach for nested sections
    start_pattern = rf'<section[^>]*identifier="{re.escape(identifier)}"[^>]*>'
    start_match = re.search(start_pattern, content)

    if not start_match:
        return f"Error: Section {identifier} not found in {xml_file.name}"

    # Find matching closing tag (handle nesting)
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

    # Convert XML to readable text
    text = xml_to_text(xml_section)
    return text


def xml_to_text(xml: str) -> str:
    """Convert USLM XML to readable text."""
    lines = []

    # Extract section heading
    sec_num = re.search(r'<num[^>]*value="([^"]+)"[^>]*>', xml)
    sec_head = re.search(
        r"<section[^>]*>.*?<heading[^>]*>(.*?)</heading>", xml, re.DOTALL
    )

    if sec_num:
        header = f"Section {sec_num.group(1)}."
        if sec_head:
            header += f" {clean_tags(sec_head.group(1))}"
        lines.append(header)
        lines.append("")

    # Process subsections recursively
    process_element(xml, lines, level=0)

    return "\n".join(lines)


def clean_tags(text: str) -> str:
    """Remove XML tags and clean up text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text.strip()


def process_element(xml: str, lines: list, level: int):
    """Process XML element and its children."""
    # Use iterative approach to find top-level subsections
    pos = 0
    while True:
        match = re.search(r'<subsection[^>]*identifier="([^"]+)"[^>]*>', xml[pos:])
        if not match:
            break

        start = pos + match.start()

        # Find the matching end tag (handle nesting)
        depth = 1
        i = start + len(match.group(0))
        while i < len(xml) and depth > 0:
            if xml[i : i + 11] == "<subsection":
                depth += 1
            elif xml[i : i + 13] == "</subsection>":
                depth -= 1
            i += 1

        end = i
        subsection_xml = xml[start:end]

        # Extract num and heading
        num_match = re.search(r'<num[^>]*value="([^"]+)"[^>]*>', subsection_xml)
        head_match = re.search(
            r"<subsection[^>]*>.*?<heading[^>]*>(.*?)</heading>",
            subsection_xml,
            re.DOTALL,
        )

        if num_match:
            indent = "  " * level
            header = f"{indent}({num_match.group(1)})"
            if head_match:
                header += f" {clean_tags(head_match.group(1))}"
            lines.append(header)

            # Extract content (paragraphs at this level)
            content_match = re.search(
                r"<content>(.*?)</content>", subsection_xml, re.DOTALL
            )
            if content_match:
                content_text = clean_tags(content_match.group(1))
                if content_text:
                    lines.append(f"{indent}  {content_text}")

        # Find paragraphs
        for para_match in re.finditer(
            r'<paragraph[^>]*identifier="([^"]+)"[^>]*>(.*?)</paragraph>',
            subsection_xml,
            re.DOTALL,
        ):
            para_num = re.search(r'<num[^>]*value="([^"]+)"', para_match.group(0))
            para_content = re.search(
                r"<content>(.*?)</content>", para_match.group(2), re.DOTALL
            )

            if para_num:
                indent = "  " * (level + 1)
                para_text = f"{indent}({para_num.group(1)})"
                if para_content:
                    para_text += f" {clean_tags(para_content.group(1))}"
                lines.append(para_text)

        pos = end


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: extract_usc_section.py <title> <section>")
        print("Example: extract_usc_section.py 26 25B")
        print("Example: extract_usc_section.py 7 2017")
        sys.exit(1)

    title = int(sys.argv[1])
    section = sys.argv[2]

    result = extract_section(title, section)
    print(result)
