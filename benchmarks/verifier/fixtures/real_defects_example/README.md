# Real defects example fixture

A two-case stand-in for `benchmarks/verifier/real_defects_v0/`, which another
session is producing. The loader in `encodebench_verifier/sources/real.py`
was written against the brief for that corpus and this fixture; when the real
README lands, reconcile any field-name differences in the loader, never in
the corpus directory.

Shape assumed per case (one JSON per case, sibling files next to it):

```
cases/<case_id>/case.json
cases/<case_id>/provision.txt     the provision text the encoder was given
cases/<case_id>/pre_fix.yaml      the defective artifact (before the repair)
cases/<case_id>/post_fix.yaml     the repaired artifact (the control; not proven clean)
```

`case.json` fields:

| field | meaning |
|---|---|
| `case_id` | stable id; the pair id on the board |
| `citation` | corpus citation path |
| `defect_kind` | one of the six synthetic kinds, or any other label (kept as `other:<label>`) |
| `locator` | `{path, rule_name, detail, token}` or a path string |
| `provision`, `pre_fix`, `post_fix` | file names relative to the case JSON (defaults shown above) |
| `hashes` | `{provision, pre_fix, post_fix}` sha256 of the file contents; verified when present |
| `generator_model` | model that produced `pre_fix` |
| `fix_reference` | where the repair is recorded (PR, run id, findings file) |
