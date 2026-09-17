"""EncodeBench verifier track: a benchmark of fidelity judges, not encoders.

The encoder track (``benchmarks/encodebench_uk_v1.yaml`` folded by
``axiom-encode eval-board``) scores models that *write* RuleSpec. This track
scores the models that *read* it: given a provision and an artifact, does the
judge notice when the artifact is wrong, and does it stay quiet when it is
right?

Two case sources are first class:

* ``synthetic`` — a versioned, seeded mutator plants one defect inside a
  known-good artifact. Every defective case ships with its unmodified original
  as the control, so ground truth is known by construction.
* ``real`` — pre-fix / post-fix artifact pairs recorded from actual repair
  rounds (``benchmarks/verifier/real_defects_v0/``, produced by another
  session). Post-fix artifacts are controls that are *not* guaranteed clean.

Judges share one interface (:mod:`encodebench_verifier.judges`), emit one
results contract (:mod:`encodebench_verifier.results`), and fold into one
board (:mod:`encodebench_verifier.board`) that refuses non-comparable inputs
the way ``harness/eval_board.py`` does.
"""

from __future__ import annotations

__version__ = "0.1.0"

SUITE_SCHEMA = "axiom-encode/encodebench-verifier-suite/v1"
RESULTS_SCHEMA = "axiom-encode/encodebench-verifier-results/v1"
BOARD_SCHEMA = "axiom-encode/encodebench-verifier-board/v1"

# The planted-defect taxonomy. Order is load-bearing: boards render kinds in
# this order and the Jev runner asks one Noul per kind in this order.
DEFECT_KINDS: tuple[str, ...] = (
    "amount_changed",
    "boundary_flipped",
    "conjunct_dropped",
    "polarity_swapped",
    "date_or_period_wrong",
    "entity_wrong",
)

DEFECT_KIND_DESCRIPTIONS: dict[str, str] = {
    "amount_changed": (
        "a number that also appears verbatim in the provision was changed "
        "inside formula or value text"
    ),
    "boundary_flipped": (
        "one comparison boundary was flipped (>= to >, > to >=, <= to <, "
        "< to <=) inside formula text"
    ),
    "conjunct_dropped": "one `and` conjunct was deleted from a formula",
    "polarity_swapped": "one `and` was turned into `or` or vice versa",
    "date_or_period_wrong": (
        "a version's effective date was moved by a year, or a rule's period "
        "was changed, where the provision states the original"
    ),
    "entity_wrong": "a rule's entity was changed to another entity",
}

# Every case in a pair carries one of these variants. ``control`` is the
# artifact believed good (the unmodified original for synthetic cases, the
# post-fix artifact for real cases); ``defective`` carries the defect.
VARIANT_CONTROL = "control"
VARIANT_DEFECTIVE = "defective"
VARIANTS: tuple[str, ...] = (VARIANT_CONTROL, VARIANT_DEFECTIVE)
