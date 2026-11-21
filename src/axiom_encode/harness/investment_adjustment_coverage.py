"""Source-owned Schedule N positive witnesses, without a capture authority grant.

Every supplied segment is re-observed from raw records. A collection of segments
is not represented as one execution. Refusal and protected-capture obligations
remain explicit until their independent consumers have supplied them.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

from .bound_investment_adjustment_operands import FLAGS, INPUT_ROWS, OUTPUT_ROWS, RATE
from .investment_adjustment_execution import PREFIX
from .investment_adjustment_suite import observe_investment_adjustment_suite


@dataclass(frozen=True)
class InvestmentAdjustmentCoverage:
    segment_companions: tuple[str, ...]
    # Role, source citation, exact source span, principal owners, record identities.
    witnesses: tuple
    missing: tuple[str, ...]
    pending: tuple[str, ...] = (
        "protected_capture_required",
        "part_c_inconsistent_input_refusal_required",
        "numeric_occurrence_allocation_required",
    )


def _name(suffix):
    return "schedule_n_" + suffix


def _out(row):
    return _name(OUTPUT_ROWS[row])


# The fixed spans refer to the fully authenticated, byte-bound parent. Distinct
# identical floor sentences retain distinct source identities.
FORMULAS = (
    (78, 850, (_out(18), _out(28), _out(36))),
    (851, 881, (_out(40),)),
    (
        1365,
        1576,
        (
            _out(36),
            _out(40),
            _name("part_c_applies"),
            _name("part_c_input_consistent"),
            _out(54),
        ),
    ),
    (1854, 1884, (_out(64),)),
    (1921, 2242, (_out(54), _out(60), _out(64), _out(76))),
    (2419, 2505, (_out(76), _out(78), _out(80))),
)


def observe_investment_adjustment_coverage(
    bound, content, parent, guide, artifact, engine_pin, segments, *, caller
):
    """Revalidate all raw segments before allocating any source-owned witnesses."""
    from .bound_investment_adjustment_operands import CARRYOVER, CONTACT, GUIDE, PART_C
    from .bound_investment_adjustment_structure import CITATION

    if type(segments) is not list or not segments:
        raise ValueError("nonempty complete raw segments required")
    suites = []
    for segment in segments:
        if type(segment) is not dict or set(segment) != {"companion", "calls"}:
            raise ValueError("raw companion and complete calls only")
        suites.append(
            observe_investment_adjustment_suite(
                bound,
                content,
                parent,
                guide,
                artifact,
                engine_pin,
                segment["companion"],
                segment["calls"],
                caller=caller,
            )
        )
    if len({s.companion_sha256 for s in suites}) != len(suites):
        raise ValueError("duplicate segment is not independent evidence")
    records = tuple(r for s in suites for r in s.records)
    if len({r.case_sha256 for r in records}) != len(records):
        raise ValueError("repeated case cannot be allocated twice")
    observations = []
    for record in records:
        # This policy has exactly one individual and no relations. Preserve the
        # raw case identity while comparing input roles across cases.
        outputs = {k.removeprefix(PREFIX): v for k, v in record.outputs}
        inputs = {name: value for entity, name, value in record.inputs}
        reached = {name: value for entity, name, value in record.reached}
        observations.append((record.case_sha256, outputs, inputs, reached))
    witnesses, missing = [], []

    def add(role, citation, span, owners, matching):
        identities = tuple(matching)
        if identities:
            witnesses.append((role, citation, span, owners, identities))
        else:
            missing.append(role)

    for start, end, owners in FORMULAS:
        add(
            f"formula:{start}:{end}",
            CITATION,
            (start, end),
            owners,
            (key for key, out, _, _ in observations if all(n in out for n in owners)),
        )

    # Full-unit roles that the generic flattened-source extractor does not emit.
    for role, principal, flag, citation, excerpt in (
        (
            "enclosure",
            _name("enclosure_required"),
            "completing_schedule_n",
            CITATION,
            "Enclose this schedule with your return.",
        ),
        (
            "records_copy",
            _name("copy_guide_reference_applies"),
            "wants_copy_for_records",
            CITATION,
            "If you want to keep a copy for your records, see the guide.",
        ),
        (
            "qualified_gain_contact",
            _name("qualified_gain_loss_contact_required"),
            FLAGS[2],
            GUIDE,
            CONTACT,
        ),
    ):
        source = parent if citation == CITATION else guide
        start = source.index(excerpt)
        pairs = []
        for a, b in combinations(observations, 2):
            changed = {n for n in a[2] if a[2][n] != b[2][n]}
            if (
                changed == {flag}
                and {a[2][flag], b[2][flag]} == {False, True}
                and principal in a[1]
                and principal in b[1]
                and a[1][principal] == a[2][flag]
                and b[1][principal] == b[2][flag]
            ):
                pairs.append((a[0], b[0]))
        add(role, citation, (start, start + len(excerpt)), (principal,), pairs)

    for row, span in ((40, (882, 917)), (60, (1818, 1853)), (64, (1885, 1920))):
        triples = []
        owners = (_out(18), _out(36), _out(54), _out(60), _out(row))
        candidates = [x for x in observations if all(n in x[1] for n in owners)]
        for triple in combinations(candidates, 3):
            changed = {
                name
                for a, b in combinations(triple, 2)
                for name in a[2]
                if a[2][name] != b[2][name]
            }
            if len(changed) != 1:
                continue
            signs = set()
            for _, out, _, _ in triple:
                raw = (
                    out[_out(18)] - out[_out(36)]
                    if row == 40
                    else out[_out(36)] - out[_out(18)]
                    if row == 60
                    else out[_out(54)] - out[_out(60)]
                )
                if out[_out(row)] != max(Decimal(0), raw):
                    raise ValueError("independent source floor differs")
                signs.add(-1 if raw < 0 else 1 if raw > 0 else 0)
            if signs == {-1, 0, 1}:
                triples.append(tuple(x[0] for x in triple))
        add(f"floor:{row}", CITATION, span, (_out(row),), triples)

    # The input is already the RO-qualified subtotal; do not claim local slip
    # classification. The static declaration and reached ordered sum bind it.
    ro = _name(INPUT_ROWS[30])
    add(
        "ro_subtotal_upstream_boundary",
        CITATION,
        (78, 850),
        (_out(36),),
        (
            key
            for key, out, inp, reached in observations
            if ro in inp and _out(36) in out
        ),
    )
    rate_keys = (
        key
        for key, out, inp, reached in observations
        if _out(28) in reached
        and _out(28) in out
        and inp[_name("resource_deduction_recovery_base")] != 0
        and out[_out(28)]
        == inp[_name("resource_deduction_recovery_base")] * Decimal("0.5")
    )
    start = parent.index("50%")
    add(
        "substantive_half_rate",
        CITATION,
        (start, start + 3),
        (_out(28), RATE),
        rate_keys,
    )
    start = guide.index(CARRYOVER)
    owners = tuple(
        _name(x)
        for x in (
            "net_investment_income_for_carryover",
            "maximum_allowable_return_252",
            "entered_return252_within_source_upper_conditions",
        )
    )
    # Upper report is never a gate on literal78. Negative raw-copy probes are
    # not lawful-deduction witnesses. Require explicit nonnegative upstream
    # facts here, while preserving their separate raw-copy diagnostic meaning.
    eligible = [
        x
        for x in observations
        if all(n in x[1] for n in owners)
        and all(not isinstance(v, Decimal) or v >= 0 for v in x[2].values())
    ]
    net = owners[0]
    for sign, suffix in ((-1, "negative"), (0, "zero"), (1, "positive")):
        add(
            "carryover_net_" + suffix,
            GUIDE,
            (start, start + len(CARRYOVER)),
            owners,
            (
                x[0]
                for x in eligible
                if (-1 if x[1][net] < 0 else 1 if x[1][net] > 0 else 0) == sign
            ),
        )
    # Both explicit deduction-presence facts matter, including entered zero.
    start_c = guide.index(PART_C)
    principal = _name("part_c_applies")
    for left, right in ((False, False), (True, False), (False, True), (True, True)):
        add(
            f"part_c_presence:{int(left)}:{int(right)}",
            GUIDE,
            (start_c, start_c + len(PART_C)),
            (principal, _name("part_c_input_consistent"), _out(54)),
            (
                key
                for key, out, inp, reached in observations
                if inp[FLAGS[0]] is left
                and inp[FLAGS[1]] is right
                and inp[_name(INPUT_ROWS[50])] == 0
                and inp[_name(INPUT_ROWS[52])] == 0
                and all(
                    n in out
                    for n in (principal, _name("part_c_input_consistent"), _out(54))
                )
                and out[principal] is (left or right)
            ),
        )
    prior = _name(INPUT_ROWS[70])
    for label, relation in (
        ("prior_limits", -1),
        ("equal_limits", 0),
        ("net_limits", 1),
    ):
        add(
            "maximum_" + label,
            GUIDE,
            (start, start + len(CARRYOVER)),
            owners,
            (
                x[0]
                for x in eligible
                if x[1][net] > 0
                and (
                    -1
                    if x[2][prior] < x[1][net]
                    else 1
                    if x[2][prior] > x[1][net]
                    else 0
                )
                == relation
            ),
        )
    entered = _name(INPUT_ROWS[78])
    for label, state in (
        ("upper_conditions_hold", True),
        ("upper_conditions_fail", False),
    ):
        add(
            label,
            GUIDE,
            (start, start + len(CARRYOVER)),
            owners,
            (x[0] for x in eligible if x[2][entered] > 0 and x[1][owners[2]] is state),
        )
    return InvestmentAdjustmentCoverage(
        tuple(s.companion_sha256 for s in suites), tuple(witnesses), tuple(missing)
    )
