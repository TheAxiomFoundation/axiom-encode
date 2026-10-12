CI now runs the six cross-repository tests in
`tests/test_core_export_integration.py` against a real `axiom-core` CLI. They
used to skip in every encoder CI run, which never set `AXIOM_CORE_BIN`. A new
`real-core` job checks out TheAxiomFoundation/axiom-core at a pinned commit
(`d989fb08`, core main, whose own interoperability CI ran these tests green),
compiles it in the run with cargo dependencies cached, and runs
`pytest -m real_core` with `AXIOM_REQUIRE_REAL_CORE=1`. Under that flag a
missing, unusable or unpinned core fails these tests instead of skipping them.
The tests get core through one `real_core` fixture (`tests/real_core.py`),
and a test that skips for a missing core without that fixture fails everywhere.
