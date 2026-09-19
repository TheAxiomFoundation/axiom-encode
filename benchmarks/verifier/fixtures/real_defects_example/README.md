# Real defects example fixture

A three-record stand-in for `benchmarks/verifier/real_defects_v0/` (axiom-encode
PR #1659), written in that corpus's own `case.json` key names so the loader in
`encodebench_verifier/sources/real.py` is tested against the schema of record:

```
cases/<id>/case.json         id, jurisdiction, repo, commit, parent_commit, module_path,
                             corpus_citation_path, corpus_release, defect_kind, other_kind,
                             confidence, description, locator{pre_fix_lines, post_fix_lines,
                             rule_names, rule_path}, pre_fix_artifact_sha256,
                             post_fix_artifact_sha256, provision_sha256, fix_stage,
                             triage_status, family_id, family_representative, artifacts_shipped
cases/<id>/pre_fix.yaml      the defective artifact (before the correcting commit)
cases/<id>/post_fix.yaml     the repaired artifact (the control; not proven clean)
cases/<id>/provision.txt     the provision text
```

`rd-0001` is a boundary correction, `rd-0002` an unrepresented clause (kept as
an `other:` kind on the board), and `rd-0003` a metadata-only family member
(`artifacts_shipped: false`) that the loader must skip and count, never invent.
