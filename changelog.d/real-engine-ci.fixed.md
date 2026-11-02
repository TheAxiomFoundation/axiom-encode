CI now runs the tests that execute the real axiom-rules-engine. They used to
skip in CI, which had no engine, and ran only where a developer's checkout
happened to exist (43 tests on main, plus the 41 that #1852 added). A new
`real-engine` job checks out the engine at a pinned commit (`89571cc2`,
rulespec-us's reviewed validation pin), builds it with `axiom-encode engine-bind`
(cargo dependencies cached, the engine itself always compiled), and runs
`pytest -m real_engine` with `AXIOM_REQUIRE_REAL_ENGINE=1`, under which a
missing, unusable or unreceipted engine fails these tests instead of skipping
them. The tests find the engine through one `real_engine` fixture
(`tests/real_engine.py`), which marks them `real_engine`, instead of hard-coded
local paths, and a test that skips for a missing engine without that marker
fails everywhere.
