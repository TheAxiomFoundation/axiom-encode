# Receipt 0.6.0 build report — axiom-encode

`receipt` moved from 0.5.1 to 0.6.0. The change is two files, `pyproject.toml`
and `uv.lock`. Everything asked for in the brief was delivered: the pin, the
regenerated lock with the 0.6.0 hashes, baseline and new-pin runs of the
repository's own checks, one commit, a push, and a draft PR.

The previous lane stopped here because the sandbox could not resolve
github.com. Network was available in this run, so the work completed.

## Why 0.6.0 is inert in this repository

0.6.0 is a breaking release: it changed the tree reader, the corpus verifier,
the release chain, the append gate, `receipt verify` and the attestation origin
parser. Commit- and tree-addressed entry points now take an entered
`TreeSnapshot`, `run_verification` takes a `LoadedSpec`, verdict objects gained
fields, and a large set of names was removed.

None of it is reachable here. `axiom-encode` uses one receipt submodule,
`receipt.sign`, at one call site — `src/axiom_encode/cli.py:46` imports it, and
`cli.py:25846-25865` (`_applied_encoding_manifest_signature_issue`, the
threshold keyring check on apply manifests) calls `KeyringSpec`, `KeySpec`,
`raw_public_key_sha256`, `verify_threshold` and `SignError`.
`tests/test_receipt_sign_adoption.py` covers that adoption.

I verified this rather than taking the brief's word for it:

1. **`receipt/sign.py` is byte-identical in 0.5.1 and 0.6.0** — sha256
   `827d9f8a508b9fed12f8c6e22cb9d108f6cd7e48c35036279d3ae068f9e7fc66` in both
   wheels and in the installed 0.6.0 environment. `diff` is empty.
2. **The import closure is two files.** `receipt/__init__.py` has zero import
   statements and differs only in `__version__`, so `import receipt.sign` loads
   `__init__.py` + `sign.py`. `sign.py` imports only stdlib and `cryptography`,
   never another receipt module — so no changed code is even loaded.
3. **The removed names are all elsewhere.** The 0.6.0 changelog enumerates
   removals for `release_chain`, `append_gate` and `corpus` only, and its 0.6.0
   section never mentions any `receipt.sign` name.
4. **AST-scoped audit** of every tracked `.py` file: `sign` is the only receipt
   submodule reached. This mattered — `receipt` is also a local dict variable
   name throughout `cli.py` (50+ sites like `receipt.get("schema_version")`), so
   a plain grep is ambiguous; the audit excludes shadowed scopes.
5. **Side-by-side load of both wheels' sign modules**: all five names have
   identical signatures, dataclass fields and MRO.
6. **Dependency metadata unchanged** (`requires-python >=3.11`,
   `cryptography>=42`), so no new transitive dependency enters the lock, and
   `uv lock` moved only the receipt entry.

## Lock hashes

Confirmed three ways — the brief, the PyPI API, and re-hashing the downloaded
artifact — and they agree with what `uv lock` recorded.

| Artifact | sha256 |
| --- | --- |
| `receipt-0.6.0-py3-none-any.whl` | `84dd540bc77f14547bcf5b4654ff22184a404aa280d8b13cda8e179593575734` |
| `receipt-0.6.0.tar.gz` | `c84f221d83099dcd86d271de8ecadf10e7eb987b7a13761af520ac9311bf773d` |

The 0.5.1 wheel re-hashed to `baacd750...e350`, matching the pre-change
`uv.lock`, so both sides of the diff are authenticated.

## Checks

Run exactly as `.github/workflows/ci.yml` runs them, on `origin/main` first and
then on this head, each in a freshly created venv.

| Command | Baseline 0.5.1 | New pin 0.6.0 |
| --- | --- | --- |
| `uv venv --python 3.13` | exit 0 | exit 0 |
| `uv pip install -e ".[dev]"` | exit 0 | exit 0 |
| `.venv/bin/python -m towncrier build --draft --version 0.0.0` | exit 0 | exit 0 |
| `.venv/bin/pytest tests/` | 12 failed, 14017 passed, 36 skipped (27:59) | 12 failed, 14017 passed, 36 skipped (17:51) |
| `ruff check src/ tests/` | exit 0 | exit 0 |
| `ruff format --check src/ tests/` | exit 0 | exit 0 |
| `uv run ruff check pyproject.toml src/axiom_encode scripts tests` | exit 0 | exit 0 |
| `python -m compileall -q src/axiom_encode scripts` | exit 0 | exit 0 |
| focused `-k "rulespec or EncoderPrompt"` | 1 failed, 2647 passed, 1990 deselected | 2648 passed, 1990 deselected |
| `pytest tests/test_receipt_sign_adoption.py` | 7 passed | 7 passed |

**The delta is zero.** `diff` of the two full-suite FAILED lists is empty — the
same 12 node ids at both pins.

I did not stop on those 12, because they are not failures of this change: they
reproduce identically on unchanged `origin/main` at the old pin. They are all
supervisor/provisioning tests requiring a root-owned system Git (9 in
`test_provision_supervisor.py`, 1 in `test_provision_verification_supervisor.py`,
2 in `test_signing_supervisor.py`), failing with
`SystemExit: trusted git path is not root-owned: /opt/homebrew/Cellar/git/2.53.0_1/bin/git`.
CI is `ubuntu-latest`, where `/usr/bin/git` is root-owned, and is green on
`main`; this is a macOS-Homebrew-only condition.

The baseline focused failure was the wall-clock assertion in
`test_rulespec_proof_reference_chain_resolution_is_bounded`
(`assert 0.050757750010234304 < 0.05` — short by 0.8 ms on a loaded machine).
It passes at the new pin and passed in both full runs, so it is flaky.

**No receipt refusal text was emitted at either pin** — receipt raised nothing,
so there is no verbatim refusal to quote.

## Notes

- `pyproject.toml` sets `testpaths = ["tests"]`, so CI's `pytest tests/` and the
  bare `uv run pytest` in `CLAUDE.md` select the identical 14065 node ids. I
  verified that with `--collect-only` instead of executing the 14k-test suite
  twice per pin.
- The brief says to keep `PROGRESS.md` untracked, but it is already tracked in
  `origin/main`. Deleting it would have added a third file to the change, so I
  kept my notes unstaged instead; the commit is the two intended files only.
- Scratch evidence is in `.lane-materials/` (gitignored): the API-surface audit,
  the extracted changelog section, both failure lists, and all check logs.
- I did not add a changelog fragment. `towncrier --draft` passes without one and
  the brief scoped the change to two files.

## Result

PR: https://github.com/TheAxiomFoundation/axiom-encode/pull/1582
Head OID: 82d3decdbb0f23331441a4b6be1758437641476f
Counts: full suite 12 failed / 14017 passed / 36 skipped at BOTH pins (identical failure sets); adoption 7 passed at both; focused 2647 passed +1 flaky failure at baseline, 2648 passed at the new pin; towncrier, ruff check, ruff format, compileall exit 0 at both.
