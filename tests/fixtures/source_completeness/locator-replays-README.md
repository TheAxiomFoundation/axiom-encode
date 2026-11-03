# Citation locator replay provenance

These committed fixtures contain candidate YAML, companion tests, source text,
diagnostics, and source metadata. Candidate and companion-test hashes are in
each `issues.json`; source hashes and corpus row identities are in
`source-metadata.json`. The replay tests verify the committed hashes and use
only committed files. The directory names and metadata retain their recorded
labels; the checkout does not independently authenticate a signing event,
original candidate archive, or historical run outcome.

* `signed_reencode_37936331759`: candidate and tests for
  `us-ok/statute/68-2906`. Its `issues.json` labels the candidate as the
  terminal `openai-gpt-6-sol` candidate and records a numeric-recall demand for
  chapter 162. The fixture contains no second candidate with which to verify
  the metadata's distinction from an earlier snapshot. Terminal session-law
  history cleanup removes the remaining history-only demands.
* `signed_reencode_37936338132`: candidate, tests, and diagnostics for
  `us-az/statute/43-1072`. Statutes at Large volume/page numbers are excluded
  from numeric recall; opening section numbers 43 and 1072 and the caption's
  apparent computations remain demanded. Real omitted table amounts and
  computations stay rejected; companion tests remain part of the fixture.
* `signed_reencode_37936341740`: candidate, tests, and diagnostics for
  `us-va/statute/58.1/58.1-322.03`. The tests require removal of history-only
  chapter demands and keep real unencoded deductions rejected.

The recorded source comparison checked all three sources against their
identified rows in corpus commit
`8f7d60aaced28ee4252b9237f9d6e02360dc34bc`, including provision-file and body
hashes. The metadata names corpus release
`us-rulespec-2026-08-08-obbb-alien-snap` and encoder version `0.2.2154`; those
are recorded labels. Replay checks run `analyze_complete_source_unit` with the
`en-US` numeric inventory and grounding extractors, matching these U.S. sources.

The replays run without a caption mask or caption heading metadata. Direct
completeness-only replay totals are recorded separately from `issues.json`,
which retains the recorded pipeline diagnostics:

| Labelled fixture | Main baseline | With both locator masks | Status |
| --- | ---: | ---: | --- |
| Oklahoma `37936331759` | 4 | 0 | Observed |
| Arizona `37936338132` | 53 | 51 | Observed |
| Virginia `37936341740` | 138 | 117 | Observed |

Arizona's earlier 53-to-47 improvement combined two Statutes at Large numeric
exclusions with four caption-related diagnostics. Removing the caption mask
restores demands for 43 and 1072, one apparent-computation output, and its
companion-test evidence; the Statutes at Large values 49 and 620 remain
excluded. These replay totals are not full production encode/apply or signing
runs. Oklahoma and Virginia keep the diagnostic reductions shown above;
both genuine-omission controls still reject.

The subdivision clause-cue mask (#1831) removes Virginia's numeric demand for
7, which came only from `this subdivision 7 if such payments ...`; the
`subdivision B 10 of § 58.1-301 related to ...` locator goes too, while the
operative `10 percent of the recognized gain` keeps the value 10 demanded.
The carryforward consequent in subdivision 7 a (#1832) no longer splits into
a second condition, which removes the `source-explicit-conditions` diagnostic
for `college_savings_account_carryforward`. The direct Virginia replay
reports 114 diagnostics.

The own-section caption check (#1827) uses only the resolver's rendered
headings, which these replays do not pass, so the totals above stand.
`tests/test_source_completeness_own_section_captions.py` authenticates
Arizona's first line as block-1's rendered heading: the text after it hashes
to the component row's committed `body_sha256`. With that heading, the
Arizona replay drops exactly the numeric demands for 43 and 1072, the
caption's apparent-computation output and its companion-test demand (51 to
47). Virginia's first line also authenticates, but `§ 58.1-322.03 .` is no
bare own-section identifier, so its diagnostics do not change.
