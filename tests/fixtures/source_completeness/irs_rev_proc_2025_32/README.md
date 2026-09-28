# IRS Rev. Proc. 2025-32 page 14 regression evidence

These fixtures preserve the supervised encoder's final rejected candidates for
[axiom-encode issue #1707](https://github.com/TheAxiomFoundation/axiom-encode/issues/1707).
The RuleSpec YAML, test YAML, and `issues.json` files are copied verbatim from the
supplied run artifacts; they are regression evidence, not hand-authored modules.

- `precise_deferral/`: artifact `36174590343`, encoder `0.2.2049`, from
  `generated/target/final-rejected-candidate/policies/irs/rev-proc-2025-32/child-tax-credit{,.test}.yaml`.
- `max_formula/`: artifact `36072174559`, encoder `0.2.2046`, from the same relative paths.
- `page-14.json`: the complete corpus record with citation path
  `us/guidance/irs/rev-proc-2025-32/page-14`, extracted from
  `data/corpus/provisions/us/guidance/2026-05-02-irs-rev-proc-2025-32-r2026-07-15-self-contained.jsonl`.

The original max-formula test artifact varies phaseout thresholds and adjusted
gross income but keeps earned income below adjusted gross income in all phaseout
cases. It does **not** contain a pair that switches which `max()` operand binds.
Regression tests must add such pairs explicitly; the original cases alone must
not satisfy the source's "or, if greater" condition.

Both original issue reports label the two source conditions `page-14(3) [Absatz 3]`.
The source corpus record instead identifies jurisdiction `us`, language `en`, and
document class `guidance`.
