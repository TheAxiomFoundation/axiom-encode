"""Independent timing oracles for the proof-location language and rule shapes."""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness.amendment_timing import amendment_timing_issues
from axiom_encode.harness.proof_validator import validate_rulespec_proofs

AMENDMENT = "nz/statute/act/public/2026/0054/section/15"
PRINCIPAL = "nz/statute/act/public/2018/0032/schedule/4/part/8"
COMMENCEMENT = "nz/statute/act/public/2026/0054/section/2"
SHAPES = ("boolean", "scalar", "table", "derived_relation", "data_relation")
OWNERS = ("rule_metadata", "rule_proof", "version_metadata", "version_proof")
SPELLINGS = (
    "indexed",
    "spaced_indexed",
    "padded_indexed",
    "trailing_dot",
    "bare",
    "spaced_bare",
    "shorthand",
    "noninteger",
    "decimal",
    "signed",
)
SOURCE_TEXTS = {
    AMENDMENT: "The replacement behavior commences on the specified date.",
    PRINCIPAL: "The principal behavior applies before its replacement.",
}
_ABSENT = object()


def _proof(path, citation=AMENDMENT):
    return {
        "atoms": [
            {
                "path": path,
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": citation,
                    "excerpt": SOURCE_TEXTS[citation],
                },
            }
        ]
    }


def _fields(shape):
    field = "values" if shape in {"table", "data_relation"} else "formula"
    return (
        "effective_from",
        "effective_to",
        field,
        f"{field}.household_size_1",
        "condition",
        "metadata",
        "",  # The version itself is also an accepted indexed location.
    )


def _path(field, spelling, index):
    suffix = f".{field}" if field else ""
    if spelling == "indexed":
        return f"versions[{index}]{suffix}"
    if spelling == "spaced_indexed":
        return f" versions [ {index} ] {suffix.replace('.', ' . ')} "
    if spelling == "padded_indexed":
        return f"versions[{index:03d}]{suffix}"
    if spelling == "trailing_dot":
        return f"versions[{index}]{suffix}."
    if spelling == "bare":
        return field or "versions"
    if spelling == "spaced_bare":
        return f" {field.replace('.', ' . ') or 'versions'} "
    if spelling == "shorthand":
        return f"versions{suffix}" if field else "versions"
    malformed_index = {"noninteger": "x", "decimal": "0.5", "signed": "+999"}
    return f"versions[{malformed_index[spelling]}]{suffix}"


def _version(shape, effective):
    result = {"effective_from": effective}
    if shape in {"table", "data_relation"}:
        result["values"] = {"household_size_1": 42}
    else:
        result["formula"] = "true" if shape == "boolean" else "42"
    return result


def _content(
    *,
    shape,
    owner,
    field,
    spelling,
    start,
    commencement,
    cutoff=_ABSENT,
    version_count=2,
    owner_index=1,
    unversioned=False,
):
    """Build cases without consulting timing's path parsers or date helpers."""
    kind = {
        "boolean": "derived",
        "scalar": "parameter",
        "table": "parameter",
        "derived_relation": "derived_relation",
        "data_relation": "data_relation",
    }[shape]
    # Rule-level shorthand explicitly names version zero. Local shorthand at
    # another owner is intentionally invalid and must retain that owner's bound.
    index = 0 if spelling == "shorthand" and owner.startswith("rule") else owner_index
    if unversioned:
        index = 0
    versions = [_version(shape, commencement) for _ in range(version_count)]
    if start is _ABSENT:
        versions[index].pop("effective_from")
    else:
        versions[index]["effective_from"] = start
    if cutoff is not _ABSENT:
        versions[index]["effective_to"] = cutoff
    rule = {"name": "replacement_eligibility", "kind": kind}
    rule["metadata"] = {"proof": _proof("effective_from", PRINCIPAL)}
    if unversioned:
        rule.update(versions[0])
    else:
        rule["versions"] = versions
    atom_owner = rule if owner.startswith("rule") else versions[index]
    proof = _proof(_path(field, spelling, index))
    if owner.endswith("metadata"):
        atom_owner.setdefault("metadata", {})["proof"] = proof
    else:
        if atom_owner is rule:
            rule.pop("metadata")
        atom_owner["proof"] = proof
    return yaml.safe_dump({"format": "rulespec/v1", "rules": [rule]}), index


def _document(commencement):
    return SimpleNamespace(
        citation_path=AMENDMENT,
        metadata={
            "commencement_date": commencement,
            "commencement_citation_path": COMMENCEMENT,
        },
    )


def _oracle_date(value):
    """Interpret generated dates independently of production helpers."""
    if type(value) is date:
        return value
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _check_case(*, cutoff=_ABSENT, **case):
    content, index = _content(cutoff=cutoff, **case)
    proof = validate_rulespec_proofs(content, source_texts=SOURCE_TEXTS)
    # A version-owned shorthand means version zero. No other generated path
    # targets a different existing version, and malformed integer spellings
    # remain conservatively accepted by base proof validation.
    mismatched_owner = (
        case["owner"].startswith("version")
        and case["owner_index"] != 0
        and case["spelling"] == "shorthand"
        and bool(case["field"])
    )
    assert proof.passed is not mismatched_owner, (content, proof.issues)

    start = case["start"]
    commencement = case["commencement"]
    parsed_start = _oracle_date(start)
    # This oracle uses the generated field and spelling, rather than parsing
    # the path with the implementation. Only a real, local, pre-commencement
    # cutoff can prove the end of principal history.
    real_cutoff_location = case["field"] == "effective_to" and case["spelling"] not in {
        "noninteger",
        "decimal",
        "signed",
    }
    parsed_cutoff = _oracle_date(cutoff)
    principal_history = (
        real_cutoff_location
        and not mismatched_owner
        and parsed_start is not None
        and parsed_cutoff is not None
        and parsed_start <= parsed_cutoff < commencement
    )
    expect_bound = (
        parsed_start is None or parsed_start < commencement
    ) and not principal_history
    issues = amendment_timing_issues(content, [_document(commencement)])
    timing = [
        issue
        for issue in issues
        if "predates commencement" in issue
        or "must declare a valid effective_from" in issue
    ]
    assert bool(timing) is expect_bound, (content, issues, principal_history)
    if expect_bound:
        assert any(f"versions[{index}]" in issue for issue in timing), issues
    return content, issues


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("owner", OWNERS)
def test_every_accepted_proof_location_keeps_amendment_bound(shape, owner):
    """Enumerate anchor/spelling families instead of hoping sampling hits them."""
    for field in _fields(shape):
        for spelling in SPELLINGS:
            _check_case(
                shape=shape,
                owner=owner,
                field=field,
                spelling=spelling,
                start=date(2018, 11, 26),
                commencement=date(2027, 4, 1),
                owner_index=1,
            )


@st.composite
def _generated_cases(draw):
    shape = draw(st.sampled_from(SHAPES))
    owner = draw(st.sampled_from(OWNERS))
    version_count = draw(st.integers(min_value=1, max_value=5))
    owner_index = draw(st.integers(min_value=0, max_value=version_count - 1))
    commencement = draw(
        st.dates(min_value=date(2025, 1, 1), max_value=date(2030, 12, 31))
    )
    start_date = draw(
        st.dates(min_value=date(2010, 1, 1), max_value=date(2035, 12, 31))
    )
    start = draw(
        st.one_of(
            st.just(start_date),
            st.just(start_date.isoformat()),
            st.none(),
            st.just(_ABSENT),
            st.sampled_from(["", "2027-02-30", "not a date", 20270401]),
        )
    )
    cutoff = draw(
        st.one_of(
            st.just(_ABSENT),
            st.none(),
            st.sampled_from(["", "not a date", "2027-02-30", 20270331]),
            st.dates(min_value=date(2000, 1, 1), max_value=date(2040, 12, 31)),
            st.dates(min_value=date(2000, 1, 1), max_value=date(2040, 12, 31)).map(
                date.isoformat
            ),
            st.just(commencement),
            st.just(commencement - timedelta(days=1)),
            st.just((commencement - timedelta(days=1)).isoformat()),
            st.just(start_date - timedelta(days=1)),
        )
    )
    return {
        "shape": shape,
        "owner": owner,
        "field": draw(st.sampled_from(_fields(shape))),
        "spelling": draw(st.sampled_from(SPELLINGS)),
        "start": start,
        "commencement": commencement,
        "cutoff": cutoff,
        "version_count": version_count,
        "owner_index": owner_index,
    }


@given(case=_generated_cases())
@settings(max_examples=180, deadline=None)
def test_generated_rule_and_version_shapes_obey_independent_timing_oracle(case):
    _check_case(**case)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("owner", ("rule_metadata", "rule_proof"))
def test_unversioned_executable_fields_keep_bound_and_real_cutoff_control(shape, owner):
    for spelling in ("bare", "spaced_bare", "noninteger", "decimal", "signed"):
        for field in _fields(shape):
            for cutoff in (_ABSENT, "2027-02-30", date(2027, 3, 31)):
                _check_case(
                    shape=shape,
                    owner=owner,
                    field=field,
                    spelling=spelling,
                    start="2018-11-26",
                    commencement=date(2027, 4, 1),
                    cutoff=cutoff,
                    owner_index=0,
                    version_count=1,
                    unversioned=True,
                )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("owner", OWNERS)
def test_genuine_principal_history_cutoff_is_exempt_for_each_owner_and_shape(
    shape, owner
):
    for spelling in SPELLINGS[:7]:
        _check_case(
            shape=shape,
            owner=owner,
            field="effective_to",
            spelling=spelling,
            start=date(2018, 11, 26),
            commencement=date(2027, 4, 1),
            cutoff=date(2027, 3, 31),
            owner_index=0,
        )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("version_owned", (False, True))
@pytest.mark.parametrize("amendment_in_metadata", (False, True))
def test_simultaneous_proof_blocks_cannot_shadow_amendment_evidence(
    shape, version_owned, amendment_in_metadata
):
    content, _ = _content(
        shape=shape,
        owner="rule_metadata",
        field="effective_from",
        spelling="bare",
        start="2018-11-26",
        commencement=date(2027, 4, 1),
        owner_index=0,
    )
    payload = yaml.safe_load(content)
    rule = payload["rules"][0]
    rule["metadata"]["proof"] = _proof("effective_from", PRINCIPAL)
    owner = rule["versions"][0] if version_owned else rule
    metadata_source = AMENDMENT if amendment_in_metadata else PRINCIPAL
    direct_source = PRINCIPAL if amendment_in_metadata else AMENDMENT
    owner.setdefault("metadata", {})["proof"] = _proof(
        "effective_from", metadata_source
    )
    owner["proof"] = _proof("effective_from", direct_source)
    content = yaml.safe_dump(payload)
    assert validate_rulespec_proofs(content, source_texts=SOURCE_TEXTS).passed
    issues = amendment_timing_issues(content, [_document(date(2027, 4, 1))])
    assert any(
        "versions[0]" in issue and "predates commencement" in issue for issue in issues
    )
