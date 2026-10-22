# Israel National Health Insurance Law section 14 replay (issue #1882)

The final rejected candidate from the targeted signed re-encode run
[38485394523](https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/38485394523)
(artifact `targeted-reencode-failure-38485394523-1`, encoder 0.2.2185, four
attempts, `repair_run_id` 38461184836) for
`il/statute/national-health-insurance-law-1994/section-14`.

* `candidate.yaml` and `candidate.test.yaml` are the artifact's
  `final-rejected-candidate` files. Their sha256 values are the
  `rulespec_sha256` and `tests_sha256` recorded in `issues.json`.
* `issues.json` holds the validator issues recorded for that candidate: all
  eleven `module.deferred_outputs` were rejected as naming no exact missing
  dependency, and the Hebrew outline was read as German `Absatz` branches.
* `source.txt` is the run workspace's corpus body for the citation, from corpus
  release `il-rulespec-2026-09-29`. Its sha256 equals the candidate's
  `module.source_verification.source_sha256` and the `resolved_text_sha256` in
  `source-metadata.json`, which is the run workspace's source attestation.

The replay tests in `tests/test_source_completeness_hebrew.py` authenticate
these bytes against those hashes and run them through
`ValidatorPipeline._complete_source_unit_issues`, the method the encoder's CI
validation calls. They do not authenticate the signing event or the historical
run outcome; the directory name and `issues.json` are the recorded labels.
