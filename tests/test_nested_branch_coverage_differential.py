"""Coverage demand equality and waiver narrowing against an isolated baseline.

No generated ownership oracle is needed: every baseline demand must remain.
Physical offsets and complete source spans prevent repeated labels or shortened
excerpts from hiding a loss, including in arbitrary Unicode source text.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import select
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CITATION = "us/statute/26/21"
BASELINE_COMMIT = "e6872987ba05ac5aebfacb3e193b36dc8570c515"


class _BaselineWorker:
    """Evaluate a complete exported package without checkout imports."""

    def __init__(self, export_root):
        self.stderr_path = export_root / "baseline-stderr.log"
        self.pid_path = export_root / "baseline-worker.pid"
        self.stderr = self.stderr_path.open("w")
        self.response_buffer = b""
        # Share test adapters, never the checkout's application modules. This
        # keeps the physical-offset and completeness comparisons identical.
        adapters = "\n\n".join(
            inspect.getsource(function)
            for function in (
                _PhysicalLabel,
                _PhysicalMatch,
                _PhysicalPattern,
                _obligations,
                _coverage_with_notice,
                _cross_references,
                _complete_source_obligations,
            )
        )
        code = textwrap.dedent(
            """
            import dataclasses
            import inspect
            import json
            import sys
            import traceback
            from pathlib import Path
            from types import SimpleNamespace

            import yaml

            package_root = Path(sys.argv[1]).resolve() / "axiom_encode"
            sys.path.insert(0, str(package_root.parent))
            from axiom_encode.harness import source_completeness as sc
            from axiom_encode.harness import validator_pipeline as vp

            def assert_package_isolation():
                for name, module in tuple(sys.modules.items()):
                    if name == "axiom_encode" or name.startswith("axiom_encode."):
                        assert module.__file__, name
                        assert Path(module.__file__).resolve().is_relative_to(
                            package_root
                        ), name
            """
        )
        code += "\n" + adapters + "\n"
        code += textwrap.dedent(
            """
            assert_package_isolation()
            print(json.dumps({"ready": True}), flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    outline = SimpleNamespace(**request["outline"])
                    if request["operation"] == "coverage":
                        result = _obligations(vp, outline)
                    elif request["operation"] == "waiver":
                        result = _coverage_with_notice(
                            vp, outline,
                            summary=request["summary"],
                            status=request["status"],
                            requested_source=request["requested_source"],
                        )
                    else:
                        assert request["operation"] == "noncoverage"
                        result = {
                            "segments": vp._top_level_source_subparagraph_segments(
                                outline.source
                            ),
                            "cross_references": _cross_references(vp, outline.source),
                            "complete_source": _complete_source_obligations(
                                vp, sc, outline
                            ),
                        }
                    assert_package_isolation()
                    response = {"result": result}
                except Exception:
                    response = {"error": traceback.format_exc()}
                print(json.dumps(response), flush=True)
            """
        )
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-u", "-c", code, str(export_root / "src")],
            cwd=export_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
        )
        try:
            self.pid_path.write_text(str(self.process.pid))
            assert self._read_response(timeout=180) == {"ready": True}
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
            raise

    def _read_response(self, *, timeout=60):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.response_buffer:
            readable, _, _ = select.select(
                [self.process.stdout], [], [], max(0, deadline - time.monotonic())
            )
            assert readable, (
                f"Baseline worker {self.process.pid} response timed out after {timeout}s\n"
                + self.stderr_path.read_text()
            )
            chunk = os.read(self.process.stdout.fileno(), 65536)
            assert chunk, self.stderr_path.read_text()
            self.response_buffer += chunk
        line, self.response_buffer = self.response_buffer.split(b"\n", 1)
        response = json.loads(line)
        assert "error" not in response, response.get("error")
        return response

    def request(self, operation, outline, **fields):
        self.process.stdin.write(
            json.dumps(
                {
                    "operation": operation,
                    "outline": {
                        "source": outline.source,
                        "citation": outline.citation,
                    },
                    **fields,
                }
            )
            + "\n"
        )
        self.process.stdin.flush()
        return self._read_response()["result"]

    def close(self):
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"Baseline worker {self.process.pid} did not exit after kill"
                        ) from None
        finally:
            self.process.stdout.close()
            self.stderr.close()
            if self.process.poll() is not None:
                self.pid_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def baseline_worker(tmp_path_factory):
    """Pin the historical baseline; explicit commits can verify newer baselines."""
    commit = os.environ.get("AXIOM_DIFFERENTIAL_BASELINE_COMMIT", BASELINE_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "Use an explicit baseline commit"
    export_root = tmp_path_factory.mktemp("nested-coverage-baseline")
    archive_path = export_root / "baseline.tar"
    with archive_path.open("wb") as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", commit, "src/axiom_encode"],
            cwd=REPO_ROOT,
            stdout=archive,
            stderr=subprocess.PIPE,
            check=True,
            timeout=60,
        )
    with tarfile.open(archive_path) as archive:
        archive.extractall(export_root, filter="data")
    archive_path.unlink()
    worker = _BaselineWorker(export_root)
    try:
        yield worker
    finally:
        worker.close()


@dataclasses.dataclass(frozen=True)
class HeadingTruth:
    label: str
    start: int
    path: tuple[str, ...] | None
    owns_signal: bool


@dataclasses.dataclass(frozen=True)
class SignalTruth:
    start: int
    path: tuple[str, ...] | None


@dataclasses.dataclass(frozen=True)
class Outline:
    source: str
    citation: str
    scope: tuple[str, ...]
    headings: tuple[HeadingTruth, ...]
    signals: tuple[SignalTruth, ...]


class _OutlineWriter:
    def __init__(self, newline: str):
        self.newline = newline
        self.parts: list[str] = []
        self.headings: list[HeadingTruth] = []
        self.signals: list[SignalTruth] = []
        self.offset = 0

    def line(self, text: str):
        self.parts.append(text + self.newline)
        self.offset += len(text + self.newline)

    def heading(self, label, path, body, *, signal=False, owns_signal=None):
        self.headings.append(
            HeadingTruth(
                label,
                self.offset,
                path,
                signal if owns_signal is None else owns_signal,
            )
        )
        if signal:
            self.signals.append(SignalTruth(self.offset, path))
        self.line(f"({label}) {body}")

    def result(self, scope=(), *, unknown_scope=False):
        citation = (
            "us-or/regulation/example/section" if unknown_scope else BASE_CITATION
        )
        if scope:
            citation += "/" + "/".join(scope)
        return Outline(
            "".join(self.parts),
            citation,
            scope,
            tuple(self.headings),
            tuple(self.signals),
        )


NOTICE_FORMATS = (
    "[({label}) {word}.]",
    "[({label}) {word}. [Effective {year}]]",
    "[({label}) {word}.] [Effective {year}]",
    "[({label}) *{word}*.]",
    "[({label}) _{word}_.]",
    "[({label}) <i>{word}.</i>]",
    "[[({label}) <em>{word}.</em>]]",
    "[({label}) **{word}**.]<sup>1</sup>",
    "[({label}) {word}.]\u00b9",
    "[({label}) {word}.] — Effective {year}",
    "[\n ({label}) {word}.\n]",
    "({label}) {word}. [Effective {year}]",
)


@st.composite
def constructed_outlines(draw, *, kind=None):
    """Generate grammar choices and retain the scope of every signal heading."""
    kind = kind or draw(
        st.sampled_from(
            [
                "lettered",
                "numeric",
                "dotted",
                "mixed_dotted",
                "unknown_marker",
                "numeric_reset",
                "borrowed_boundary",
                "inline_boundary",
                "inline_heading",
                "combined",
                "repeated",
                "repeal",
                "editorial",
                "gap",
                "rate",
            ]
        )
    )
    writer = _OutlineWriter(draw(st.sampled_from(["\n", "\r\n"])))
    number = draw(st.integers(min_value=2, max_value=20))
    letter = draw(st.sampled_from(list("abcfkl")))
    gap = draw(st.sampled_from(["", " ", "\t"]))
    indentation = draw(st.sampled_from(["", "    ", "\t", "\u00a0", "\u200b"]))
    cue = draw(st.sampled_from(["means", "is eligible for", "is ineligible for"]))
    rate = draw(st.integers(min_value=11, max_value=39))
    signal = f'"benefit" {cue} an amount payable at {rate} percent.'
    cited_rate = (
        "The applicable percentage determined under section 24 "
        f"applies at {rate} percent."
    )

    if kind == "mixed_dotted":
        # An isolated dotted root may be rejected by the recognizer. A prior
        # numeric root cannot establish ownership across that uncertain boundary.
        marker, outer = draw(
            st.sampled_from([("D.", "d"), ("II.", "ii"), ("Section 2.", "2")])
        )
        marker, outer = draw(
            st.one_of(
                st.just((marker, outer)),
                st.sampled_from(
                    [
                        (f"{number + 1}.", str(number + 1)),
                        (f"{number + 1}.1.", f"{number + 1}.1"),
                        (f"§ {number + 1}.", str(number + 1)),
                        (f"Section {number + 1}", str(number + 1)),
                        (f"（{number + 1}）", str(number + 1)),
                        (f"({number + 1}]", str(number + 1)),
                        (f"[{number + 1})", str(number + 1)),
                        (f"({number + 1}", str(number + 1)),
                    ]
                ),
            )
        )
        marker = draw(
            st.sampled_from(
                [
                    marker,
                    f"*{marker}*",
                    f"**{marker}**",
                    f"<b>{marker}</b>",
                    f"[{marker}]",
                    f"[[{marker}]]",
                    f"\u200b{marker}",
                    f"\u2060{marker}",
                ]
            )
        )
        if draw(st.booleans()):
            marker = marker.translate(
                str.maketrans(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
                    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９．",
                )
            )
        marker = draw(st.text(alphabet="#>•{}:+~!\"'", min_size=0, max_size=4)) + marker
        writer.line(f"({number}) Prior numbered material.")
        writer.line(f"{marker} Definitions:")
        writer.heading(letter, (outer, letter), signal, signal=True)
        writer.line(f"{indentation}(1) {cited_rate}")
        return writer.result((outer,) if draw(st.booleans()) else ())

    if kind == "unknown_marker":
        writer.line(f"({number}) Prior numbered material.")
        boundary = draw(st.sampled_from(["(AA)", "[(D)]", "(Β)", "[(Ａ)]"]))
        writer.line(f"{indentation}{boundary} Definitions:")
        # An unfamiliar alphabetic marker does not prove any ancestor for
        # the real signal-bearing successor, even in an otherwise numeric list.
        writer.heading(letter, None, signal, signal=True)
        writer.line(f"{indentation}(1) {cited_rate}")
        return writer.result()

    if kind == "inline_heading":
        marker, outer = draw(
            st.sampled_from(
                [
                    ("D.", "d"),
                    ("E.", "e"),
                    (f"{number + 1}.", str(number + 1)),
                    (f"{number + 1}:", str(number + 1)),
                    (f"Section {number + 1}", str(number + 1)),
                    (f"§ {number + 1}", str(number + 1)),
                ]
            )
        )
        marker = draw(
            st.sampled_from(
                [
                    marker,
                    f"**{marker}**",
                    f"<i>{marker}</i>",
                    f"[{marker}]",
                    "\u2060" + marker,
                ]
            )
        )
        if draw(st.booleans()):
            marker = marker.translate(
                str.maketrans(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:",
                    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９．：",
                )
            )
        writer.line(f"({number}) Prior numbered material.")
        heading_gap = draw(st.sampled_from(["", " "]))
        writer.heading(
            "a", (str(number), "a"), "Details. " + marker + heading_gap + "Definitions:"
        )
        # The writer assigns the successor to the new inline heading before
        # either recognizer sees the source. Typography cannot prove the old
        # numeric root still owns this operative branch.
        writer.heading("c", (outer, "c"), signal, signal=True)
        return writer.result((outer,))

    if kind == "numeric_reset":
        writer.line(f"({number}) Prior numbered material.")
        writer.heading("a", (str(number), "a"), "Details.")
        reset = draw(st.sampled_from(["1", "1a"]))
        scope = (str(number), "a", reset) if draw(st.booleans()) else (reset,)
        writer.line(f"({reset}) Definitions:")
        # The requested subtree explicitly owns this reset. A parser's flat
        # fallback cannot prove it is still under the previous numeric root.
        writer.heading(letter, scope + (letter,), signal, signal=True)
        writer.line(f"{indentation}(1) {cited_rate}")
        return writer.result(scope)

    if kind == "borrowed_boundary":
        writer.heading("a", ("a",), "Administrative details.")
        writer.line("D. Terms:")
        inline = draw(st.booleans())
        if inline:
            boundary = draw(
                st.sampled_from(
                    [
                        "E. Definitions:",
                        "**E.** Definitions:",
                        "<i>E.</i> Definitions:",
                        "Ｅ． Definitions:",
                        "6: Definitions:",
                        "６： Definitions:",
                        "6.Definitions:",
                        "6. Definitions:",
                        "**6.**Definitions:",
                        "<i>6.</i>Definitions:",
                        "６．Definitions:",
                        "Section 6 Definitions:",
                        "[(AA) Definitions:]",
                        "[(6.1) Definitions:]",
                        "[(<i>E</i>) Definitions:]",
                        "（6） Definitions:",
                    ]
                )
            )
            if draw(st.booleans()):
                writer.signals.append(SignalTruth(writer.offset, None))
                writer.line(
                    f"(1) Administrative item. {boundary} {signal} {cited_rate}"
                )
            else:
                writer.line(f"(1) Administrative item. {boundary}")
                writer.signals.append(SignalTruth(writer.offset, None))
                writer.line(signal + " " + cited_rate)
            return writer.result()
        writer.line("(1) Administrative item.")
        marker = draw(
            st.sampled_from(["2.", "*D.*", "<i>D.</i>", "Section 2", "（2）", "(2]"])
        )
        prefix = draw(st.text(alphabet="#>•{}:+~!", min_size=0, max_size=4))
        if draw(st.booleans()):
            notice = draw(st.sampled_from(NOTICE_FORMATS)).format(
                label="b", word="Repealed", year=draw(st.integers(2020, 2026))
            )
            writer.line(notice)
        else:
            writer.line(f"{prefix}{marker} New outline:")
        # This later outline is deliberately unscoped. The cue cannot be
        # attributed to the earlier numbered item just because it was parsed.
        writer.signals.append(SignalTruth(writer.offset, None))
        writer.line(signal + " " + cited_rate)
        return writer.result()

    if kind == "inline_boundary":
        writer.line(f"({number}) Definitions:")
        boundary_label = draw(
            st.sampled_from(
                [
                    str(number + 1),
                    f"{number + 1}.1",
                    f"{number + 1}\u2013{number + 2}",
                    f"{number + 1}a",
                    f"<i>{number + 1}</i>",
                ]
            )
        )
        notice = draw(st.sampled_from(NOTICE_FORMATS)).format(
            label=boundary_label,
            word=draw(st.sampled_from(["Repealed", "Omitted"])),
            year=draw(st.integers(2020, 2026)),
        )
        writer.heading("a", (str(number), "a"), "Details. " + notice)
        # A skipped numbered notice after another heading can change the
        # parent. The outline deliberately leaves the successor's owner unknown.
        writer.heading("c", None, signal, signal=True)
        return writer.result()

    if kind == "lettered":
        writer.heading("a", ("a",), "Calculation:", owns_signal=True)
        writer.line(f"{indentation}(1) Terms:")
        writer.signals.append(SignalTruth(writer.offset, ("a", "1", "i")))
        writer.line(f"{indentation}(i) {signal} {cited_rate}")
        writer.heading("f", ("a", "1", "f"), signal, signal=True)
        writer.heading("b", ("b",), "Effective date.")
        return writer.result()

    if kind in {"numeric", "dotted"}:
        outer = ("d",) if kind == "dotted" else ()
        if outer:
            writer.line("D. Definitions:")
        printed_number = str(number).translate(
            str.maketrans(
                "0123456789",
                draw(
                    st.sampled_from(
                        [
                            "0123456789",
                            "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19",
                            "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
                        ]
                    )
                ),
            )
        )
        writer.line(f"{indentation}({printed_number}) Terms:")
        parent = outer + (str(number),)
        scoped = draw(st.booleans())
        scope = parent if scoped else ()
        writer.heading(letter, parent + (letter,), signal, signal=True)
        successor = chr(ord(letter) + 1)
        # A letter owns this numbered/roman list; flattening its indentation
        # must not change the following alphabetic sibling's ownership.
        writer.heading(
            successor, parent + (successor,), "Calculation:", owns_signal=True
        )
        writer.line(f"{indentation}(1) {cited_rate}")
        writer.signals.append(
            SignalTruth(writer.offset, parent + (successor, "1", "i"))
        )
        writer.line(f"{indentation}(i) {signal}")
        writer.line(f"({number + 1}) Effective date.")
        return writer.result(scope)

    if kind in {"combined", "repeated"}:
        writer.heading("a", ("a",), "Administrative details.")
        if kind == "combined":
            writer.heading("b", ("b",), f"(1){gap}(A) {signal}", signal=True)
        else:
            writer.heading("a", ("a",), f"({number}){gap}(A) {signal}", signal=True)
        writer.heading("c", ("c",), "Effective date.")
        return writer.result()

    if kind in {"repeal", "editorial", "gap"}:
        numeric_parent = kind == "repeal" and draw(st.booleans())
        parent = (str(number),) if numeric_parent else ()
        if parent:
            writer.line(f"{indentation}({number}) Terms:")
        if kind in {"repeal", "gap"}:
            writer.heading("a", parent + ("a",), "Administrative terms.")
            writer.heading("b", parent + ("b",), "Calculation:")
            writer.line("(1) Item.")
            roman = draw(st.sampled_from(["i", "ii", "iv", "vi", "x"]))
            writer.line(f"({roman}) Details.")
        if kind != "gap":
            disposition = draw(st.sampled_from(["Repealed", "Omitted"]))
            notice = draw(st.sampled_from(NOTICE_FORMATS)).format(
                label="c", word=disposition, year=draw(st.integers(2020, 2026))
            )
            writer.line(indentation + notice)
        # An outline can omit an alphabetic sibling altogether. Its following
        # root remains real even when a flattened recognizer retains stale
        # numbered/roman ancestry; an editorial notice is not required.
        successor = draw(st.sampled_from(["d", "e", "f", "j", "k", "l", "m", "r"]))
        writer.heading(successor, parent + (successor,), signal, signal=True)
        unknown = draw(st.booleans())
        scoped = numeric_parent and draw(st.booleans())
        return writer.result(parent if scoped else (), unknown_scope=unknown)

    writer.heading("a", ("a",), "Administrative details.")
    writer.line("D. Calculation:")
    writer.line(f"(1) {cited_rate}")
    writer.signals.append(SignalTruth(writer.offset, ("d", "1", "i")))
    writer.line(f"(i) {signal}")
    return writer.result()


class _PhysicalLabel(str):
    """Carry a marker's offset without changing its string value."""

    def __new__(cls, value, start):
        label = super().__new__(cls, value)
        label.start = start
        return label

    def lower(self):
        return _PhysicalLabel(super().lower(), self.start)


class _PhysicalMatch:
    def __init__(self, match):
        self.match = match

    def group(self, *groups):
        value = self.match.group(*groups)
        if groups == ("label",):
            return _PhysicalLabel(value, self.match.start())
        return value

    def __getattr__(self, name):
        return getattr(self.match, name)


class _PhysicalPattern:
    def __init__(self, pattern):
        self.pattern = pattern

    def finditer(self, *args, **kwargs):
        return (
            _PhysicalMatch(match) for match in self.pattern.finditer(*args, **kwargs)
        )

    def __getattr__(self, name):
        return getattr(self.pattern, name)


def _obligations(module, outline):
    function = module._high_signal_top_level_subparagraphs
    arguments = (
        {"corpus_citation_path": outline.citation}
        if "corpus_citation_path" in inspect.signature(function).parameters
        else {}
    )
    raw_demands = function(outline.source, **arguments)
    pattern = module._TOP_LEVEL_SOURCE_SUBPARAGRAPH_PATTERN
    try:
        module._TOP_LEVEL_SOURCE_SUBPARAGRAPH_PATTERN = _PhysicalPattern(pattern)
        measured_demands = function(outline.source, **arguments)
    finally:
        module._TOP_LEVEL_SOURCE_SUBPARAGRAPH_PATTERN = pattern
    assert measured_demands == raw_demands
    # A repeated label can have identical text at an earlier excluded heading.
    # Preserve its actual regex-match origin instead of searching by text prefix.
    result = {}
    for label, text in measured_demands:
        assert isinstance(label, _PhysicalLabel)
        assert outline.source[label.start :].startswith(f"({label})")
        result[label.start] = (str(label), text)
    assert len(result) == len(raw_demands)
    return result


def test_physical_obligations_identify_every_duplicate_by_its_offset():
    cue = '"benefit" means the amount payable.'
    source = f"(5) Prior numbered material.\n(a) {cue}\n§ 6 Definitions:\n(a) {cue}\n"
    outline = Outline(source, BASE_CITATION, (), (), ())
    first = source.index("(a)")
    second = source.rindex("(a)")
    assert _obligations(vp, outline) == {
        first: ("a", cue + "\n§ 6 Definitions:"),
        second: ("a", cue),
    }


def _gap_example():
    writer = _OutlineWriter("\n")
    writer.heading("a", ("a",), "Administrative terms.")
    writer.heading("b", ("b",), "Calculation.")
    writer.line("(1) Item.")
    writer.line("(i) Details.")
    writer.heading("d", ("d",), '"benefit" means the amount payable.', signal=True)
    return writer.result()


def _mixed_dotted_example():
    writer = _OutlineWriter("\n")
    writer.line("(5) Prior numbered material.")
    writer.line("D. Definitions:")
    writer.heading("c", ("d", "c"), '"benefit" means the amount payable.', signal=True)
    return writer.result(("d",))


def _inline_heading_examples():
    scoped = _OutlineWriter("\n")
    scoped.line("(5) Prior numbered material.")
    scoped.heading("a", ("5", "a"), "Details. D. Definitions:")
    scoped.heading("c", ("d", "c"), '"benefit" means the amount payable.', signal=True)
    borrowed = _OutlineWriter("\n")
    borrowed.heading("a", ("a",), "Administrative details.")
    borrowed.line("D. Terms:")
    borrowed.line("(1) Administrative item. E. Definitions:")
    borrowed.signals.append(SignalTruth(borrowed.offset, None))
    borrowed.line('"benefit" means the amount payable.')
    colon = _OutlineWriter("\n")
    colon.line("(5) Prior numbered material.")
    colon.line("6: Definitions:")
    colon.heading("c", ("6", "c"), '"benefit" means the amount payable.', signal=True)
    return scoped.result(("d",)), borrowed.result(), colon.result(("6",))


@st.composite
def unconstrained_outlines(draw):
    """Mix unrestricted Unicode into and around physical outline markers."""
    scope = draw(st.sampled_from([(), ("5",), ("a", "2", "i")]))
    citation = draw(
        st.sampled_from([BASE_CITATION, "us-or/regulation/example/section"])
    )
    if scope:
        citation += "/" + "/".join(scope)
    if draw(st.booleans()):
        source = draw(st.text(max_size=800))
    else:
        fragments = draw(st.lists(st.text(max_size=100), min_size=6, max_size=6))
        newline = draw(st.sampled_from(["\n", "\r\n"]))
        source = newline.join(
            [
                fragments[0],
                "(5) " + fragments[1],
                '(a) "benefit" means the amount payable. ' + fragments[2],
                "(" + fragments[3] + ") <" + fragments[4] + ">",
                "(2) " + fragments[5],
                "(i) Repealed.",
                '(b) "payment" is eligible for an amount.',
            ]
        )
    return Outline(source, citation, scope, (), ())


@pytest.mark.parametrize(
    "source",
    [
        '(5) Definitions:\n(a) "benefit" means the amount payable.\n',
        "(a) Administrative details.\nD. Calculation:\n"
        '(1) "benefit" means the amount payable.\n',
        '(5) Definitions:\n(a) "benefit" means the amount payable.\n'
        "(6) Effective date.\n",
        '(a) "benefit" means the amount payable.\nD. Calculation:\n'
        "(1) Administrative item.\n",
    ],
    ids=["numeric-ancestry", "borrowed-signal", "numeric-span", "dotted-span"],
)
def test_nested_and_borrowed_demands_equal_pinned_baseline(baseline_worker, source):
    outline = Outline(source, BASE_CITATION, (), (), ())
    baseline = {
        int(offset): tuple(demand)
        for offset, demand in baseline_worker.request("coverage", outline).items()
    }
    assert baseline, "The regression must exercise a baseline demand"
    assert _obligations(vp, outline) == baseline


@given(outline=st.one_of(unconstrained_outlines(), constructed_outlines()))
@example(outline=_gap_example())
@example(outline=_mixed_dotted_example())
@example(outline=_inline_heading_examples()[0])
@example(outline=_inline_heading_examples()[1])
@example(outline=_inline_heading_examples()[2])
@settings(max_examples=500, deadline=None, derandomize=True)
def test_generated_coverage_demands_equal_pinned_baseline(
    baseline_worker,
    outline,
):
    baseline = {
        int(offset): tuple(demand)
        for offset, demand in baseline_worker.request("coverage", outline).items()
    }
    head = _obligations(vp, outline)
    assert head == baseline, (outline, baseline, head)


def _coverage_with_notice(module, outline, *, summary, status, requested_source):
    """Observe waiver returns even when scanning would find no demands."""
    result = {}
    for key, module_status in (("required", "encoded"), ("with_notice", status)):
        content = yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {
                    "status": module_status,
                    "summary": summary,
                    "source_verification": {"corpus_citation_path": outline.citation},
                },
                "rules": [],
            }
        )
        original = module._high_signal_top_level_subparagraphs
        scanned = False

        def record_scan(*args, **kwargs):
            nonlocal scanned
            scanned = True
            return original(*args, **kwargs)

        module._high_signal_top_level_subparagraphs = record_scan
        try:
            result[key] = module.find_source_subparagraph_coverage_issues(
                content,
                source_texts={outline.citation: outline.source},
                requested_source=requested_source,
            )
        finally:
            module._high_signal_top_level_subparagraphs = original
        result[key + "_scanned"] = scanned
    # A source-missing return also skips scanning. Only an active control that
    # reached the scanner proves that the deferred module took a waiver path.
    result["waived"] = result["required_scanned"] and not result["with_notice_scanned"]
    return result


@pytest.mark.parametrize(
    "source,summary",
    [
        ("(3) Repealed.", "Administrative details."),
        ("(3) Omitted.", "(3) Omitted."),
        ("[(3) Repealed.]", "[(3) Repealed.]"),
        (
            '(a) "benefit" means the amount payable.\n(3) Omitted.\n',
            "(3) Omitted.",
        ),
        (
            '(a) "benefit" means the amount payable.\n(3) Repealed.\n',
            "[(3) Repealed.]",
        ),
        (
            '(a) "benefit" means the amount payable.\n(3) Repealed.\n',
            "(3) ...",
        ),
    ],
    ids=[
        "source-only",
        "omitted-source",
        "formatted-source",
        "omitted-summary",
        "formatted-summary",
        "ellipsis-summary",
    ],
)
def test_repeal_waiver_cannot_expand_baseline_acceptance(
    baseline_worker, source, summary
):
    outline = Outline(source, BASE_CITATION, (), (), ())
    arguments = {
        "summary": summary,
        "status": "deferred",
        "requested_source": BASE_CITATION + "/a/3",
    }
    baseline = baseline_worker.request("waiver", outline, **arguments)
    head = _coverage_with_notice(vp, outline, **arguments)
    assert head["required"] == baseline["required"]
    assert head["required_scanned"] and baseline["required_scanned"]
    if head["waived"]:
        assert baseline["waived"], (summary, baseline, head)


def test_repeal_waiver_additional_authoritative_condition_refuses_missing_tombstone(
    baseline_worker,
):
    source = (
        '(a) "benefit" means the amount payable.\n'
        "(2) Administrative details.\n(i) Operative provision.\n"
    )
    outline = Outline(source, BASE_CITATION, (), (), ())
    arguments = {
        "summary": "(i) Repealed.",
        "status": "deferred",
        "requested_source": BASE_CITATION + "/a/2/i",
    }
    baseline = baseline_worker.request("waiver", outline, **arguments)
    head = _coverage_with_notice(vp, outline, **arguments)
    assert baseline["required"]
    assert baseline["with_notice"] == []
    assert head["required"] == baseline["required"]
    assert head["with_notice"] == head["required"]
    assert baseline["waived"]
    assert not head["waived"]


@given(
    outline=st.one_of(unconstrained_outlines(), constructed_outlines()),
    summary=st.one_of(
        st.text(max_size=300),
        st.sampled_from(
            ["(i) Repealed.", "(i) Omitted.", "[(i) Repealed.]", "(i) ..."]
        ),
    ),
    status=st.sampled_from(["deferred", "entity_not_supported"]),
    requested_source=st.sampled_from(
        [None, BASE_CITATION + "/a/2/i", BASE_CITATION + "/a/3"]
    ),
)
@settings(max_examples=500, deadline=None, derandomize=True)
def test_generated_waiver_implies_pinned_baseline_waiver(
    baseline_worker, outline, summary, status, requested_source
):
    arguments = {
        "summary": summary,
        "status": status,
        "requested_source": requested_source,
    }
    baseline = baseline_worker.request("waiver", outline, **arguments)
    head = _coverage_with_notice(vp, outline, **arguments)
    assert head["required"] == baseline["required"], (outline, baseline, head)
    assert head["required_scanned"] == baseline["required_scanned"]
    if head["waived"]:
        assert baseline["waived"], (outline, summary, baseline, head)


def _cross_references(module, source):
    return module.ValidatorPipeline._extract_cross_reference_numeric_obligations(
        None, source, title="26"
    )


def _complete_source_obligations(module, complete_source_module, outline):
    content = yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []})
    result = complete_source_module.analyze_complete_source_unit(
        content,
        outline.source,
        corpus_citation_path=outline.citation,
        test_cases=[],
        extract_numeric_occurrences=module.extract_typed_numeric_inventory_occurrences_from_text,
        extract_numeric_grounding_occurrences=module.extract_typed_numeric_occurrences_from_text,
        extract_named_scalars=module.extract_named_scalar_occurrences,
        numeric_value_is_grounded=module.numeric_value_is_grounded,
    )
    # Compare every completeness finding, branch, and numeric occurrence count,
    # including numeric recall, without sharing either implementation's results.
    # Branch wording is not an obligation: U.S. findings now label a branch
    # `us/x(a)(1) [paragraph (a)(1)]` where the baseline said `us/x(a)
    # [Absatz a, Nummer 1]`, and branches carry a display-only printed-case
    # path. Both sides reduce a rendered branch citation to its path.
    import re as _re

    def _branch_path(match):
        if match.group("german"):
            parts = [
                (
                    "satz-" + component.split()[-1]
                    if component.strip().startswith("Satz")
                    else component.split()[-1].lower()
                )
                for component in match.group("german").split(",")
            ]
        else:
            parts = [
                part.lower()
                for part in _re.findall(r"\(([^()\s]+)\)", match.group("fragment"))
            ] + [
                "satz-" + component.split()[-1]
                for component in match.group("english").split(",")
                if component.strip().startswith("sentence ")
            ]
        return f"{match.group('citation')}<{'/'.join(parts)}>"

    rendered_branch = _re.compile(
        r"(?P<citation>[^\s()\[\]]+?)(?:\([^()\s]+\) "
        r"\[(?P<german>(?:Absatz|Nummer|Buchstabe|Satz) [^\]]*)\]|"
        r"(?P<fragment>(?:\([^()\s]+\))+) \[(?P<english>(?:paragraph|"
        r"subsection|subparagraph|clause|subclause|item|subitem|subdivision|"
        r"provision|sentence) [^\]]*)\])"
    )
    payload = dataclasses.asdict(result)
    for branch in payload["branches"]:
        branch.pop("display_path", None)
    payload["issues"] = tuple(
        rendered_branch.sub(_branch_path, issue) for issue in payload["issues"]
    )
    return payload


@given(outline=constructed_outlines())
@example(outline=_inline_heading_examples()[0])
@example(outline=_inline_heading_examples()[1])
@example(outline=_inline_heading_examples()[2])
@example(
    outline=Outline(
        "(a) Administrative details.\nD. Calculation:\n"
        "(1) The applicable percentage determined under section 24 applies.\n",
        BASE_CITATION,
        (),
        (HeadingTruth("a", 0, ("a",), False),),
        (),
    )
)
@settings(max_examples=500, deadline=None, derandomize=True)
def test_generated_noncoverage_obligations_are_identical_to_main(
    baseline_worker, outline
):
    # Every consumer keeps byte-identical spans, including dotted-heading text.
    # The U.S. Code outline restoration (flattened run-in markers, bracketed
    # repeal notes, subclause levels) deliberately changes U.S. Code
    # recognition and has its own tests in test_source_completeness_us_structure;
    # this pinned comparison guards everything else.
    with mock.patch.object(sc, "_us_code_outline_levels", lambda _path: None):
        complete_source = _complete_source_obligations(vp, sc, outline)
    head = {
        "segments": vp._top_level_source_subparagraph_segments(outline.source),
        "cross_references": _cross_references(vp, outline.source),
        "complete_source": complete_source,
    }
    # JSON changes tuple containers to lists on both sides without truncating
    # any source span, branch, finding, or numeric occurrence count.
    assert json.loads(json.dumps(head)) == baseline_worker.request(
        "noncoverage", outline
    )
