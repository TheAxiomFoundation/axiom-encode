# Producer lineage preflight

Status: implemented verification primitives and diagnostics; no admission authority.
This follows Max's [enrolled-producer decision](https://github.com/TheAxiomFoundation/axiom-encode/pull/1629#issuecomment-5732652158)
and the approved [notary design v33](https://github.com/TheAxiomFoundation/axiom-encode/blob/6ec1ea7785425c9beebdd8e352096217a439b9df/docs/notary-admission-design.md).
It builds on the canonicalization and tree-manifest primitives in
[#1511](https://github.com/TheAxiomFoundation/axiom-encode/pull/1511).

## Implemented boundary

This slice supplies the public-key registry and authenticated-lineage checks
needed before enrollment can produce usable evidence. It does not provision a
producer, sign arbitrary supplied files, grant merge authority, or modify the
current RuleSpec guard. Personal Codex generation remains on the supervised
runtime. Producer and actor keys stay on that host; the notary remains an
external typed signer.

The implementation provides:

1. A fail-closed enrollment-eligibility check against GitHub's current
   collaborator permission endpoint for the target repository. Only write
   (including maintain) or admin permission qualifies an operator to request
   enrollment. Read/triage, contributors without write access, and unavailable
   permission evidence do not qualify. The operator identity must come from an
   authenticated enrollment session or protected workflow in production;
   typing another person's username into a local diagnostic proves nothing
   about the caller. This is an onboarding check, not a new notary schema.
2. Strict v33 registry parsing: closed role map, canonical Ed25519 SPKI bytes,
   matching fingerprints, sorted unique entries, and total key separation
   across all seven roles and both frozen legacy roots. The notary root must
   match the caller's pinned consumer identity. No environment trust roots.
3. Detached-signature verification using the exact v2 frame and v33 role table,
   including actor plus correction-review countersignatures. A sidecar's
   fingerprint selects only a key already registered for the required role.
4. Closed generation/correction schemas, including every v33 per-draw field,
   canonical decimal values, transition endpoints and modes, sorted arrays,
   and byte-for-byte canonical JSON. Runtime identity and source metadata
   remain authenticated declarations, not measured runtime attestations.
5. Deterministic lineage-store classification under §§2.6/3.2: inherited files
   are immutable inert history; new malformed records are enumerated as
   ineligible rather than vetoing the candidate; records are bound to lane,
   epoch, protected paths, and required signature roles. All applicable
   reason codes are retained in sorted order.
6. A read-only diagnostic command over explicit Git base/subject commits. It
   reads the registry and path policy from the base, rather than the checkout
   or candidate, and reuses #1511's whole-tree checks. Its output identifies
   eligible/ineligible records and explicitly states that admission was not
   evaluated. It does not emit the normative notary pass report or receipt.

## Commands

Query enrollment eligibility using the GitHub CLI's existing authentication:

```bash
uv run python -m axiom_encode.notary.enrollment \
  --repository TheAxiomFoundation/rulespec-us --operator YOUR_GITHUB_LOGIN
```

The command uses GitHub's [repository-permission endpoint](https://docs.github.com/en/rest/collaborators/collaborators#get-repository-permissions-for-a-user),
whose base `permission` maps maintain to write and triage to read. It never
invites a contributor or changes a role. A writer receives
`eligible_to_request_enrollment: true`, `enrolled: false`, and
`reason: "custodian-authorization-required"`. Exit 0 means permission to
request enrollment; exit 1 means ineligible or unavailable evidence. This
response is a local diagnostic, not a signed credential or an authenticated
assertion about the person running the command.

For a fixture repository or a future lane containing a committed v33 registry
and path policy, inspect lineage under explicit public identities:

```bash
uv run python -m axiom_encode.notary.preflight \
  --repository /path/to/lane-checkout \
  --base "$BASE_COMMIT" --subject "$SUBJECT_COMMIT" \
  --lane TheAxiomFoundation/rulespec-nz --epoch-sha256 "$EPOCH_SHA256" \
  --notary-spki-sha256 "$NOTARY_SPKI_SHA256" \
  --legacy-apply-root "$LEGACY_APPLY_SPKI_SHA256" \
  --legacy-eval-root "$LEGACY_EVAL_SPKI_SHA256"
```

Base and subject are full 40-character commit ids. The three key identities
are public SPKI digests, never private keys. No model call or signing request
occurs. Uncommitted checkout contents are ignored. The JSON always includes
`admission: "not-evaluated"` and
`authority: "caller-supplied-base-and-pins"`. Exit 0 means classification ran,
even if records are ineligible; exit 1 means an input/structural/policy refusal.
Neither exit code is an admission verdict. The command is deliberately not
wired into RuleSpec CI or any required check.

The module name `preflight` denotes this lineage diagnostic only. It does not
implement the complete v33 trusted preflight: in particular, whole-tree
validation is not the protected-executable/intermediate-mode wall, and
trust-surface changes still need their separate transition-only refusal.

## Trust and integration

Registry bodies use `schema`, `lane`, and the seven top-level role names from
v33 §5. No new enrollment policy or runtime-identity allowlist is added to that
schema. Public verification of a key in a caller-selected registry does not
prove administrative authorization: production integration must supply a base,
epoch, and consumer/legacy pins from the authenticated finalized chain.
The local diagnostic accepts those identities as explicit inputs, labels its
result accordingly, and can never satisfy a required admission check.

Enrollment still requires the host custodian to establish the emission
boundary and Max's administrative authorization through the ceremony/genesis
or predecessor-root transition. This library verifies lineage under a given
registry; it cannot prove that a host is isolated or that a producer honestly
declared its runtime/model metadata. Removing a producer from an effective
registry prevents its signatures from authenticating new lineage. Historical
base-present files are not reauthenticated against rotated keys.

The intended contributor set is the team's authorized repository writers,
including the contributors named in Max's access announcement. Names are not
hardcoded: current custodian-granted repository permissions govern eligibility.
An invited repository collaborator can qualify without organization membership;
GitHub's "outside collaborator" label is not itself a refusal. Write access is
necessary for enrollment and authenticated contributor intake, but does not
replace producer enrollment, valid evidence, or the usual quality gates. A
random contributor without write access cannot acquire eligible producer
status merely by running the encoder with a personal Codex subscription. Production intake
must recheck the authenticated contributor's current access and tie it to the
approved host enrollment before accepting a submission; author strings in
Git commits and self-declared operator names are not authentication. Loss of
write access denies new intake and triggers administrative producer-revocation
handling; this PR does not install that lifecycle service.

The [contributor implementation package](notary-contributor-package.md) adds
coverage replay, report reconciliation, administrative transitions, chain
reconstruction, typed signing and publication plans. This diagnostic intentionally
does not invoke those admission operations or decide whether eligible records
cover the candidate's changed files. Production deployment and CI integration
are still required.
All v33 §9 preconditions remain required before anything admission-capable
merges; US activation also remains a separate rollout after the NZ pilot.

| Max's enrollment package | Supplied here | Remaining integration |
| --- | --- | --- |
| Who authenticates a producer | Current GitHub write-eligibility check and strict registered-key verification | Authenticated operator session, host evidence, Max's administrative authorization |
| Supervised runtime boundary | Exact authenticated generation/correction bodies and all v33 per-draw fields | Protected emission service, runtime measurement and host controls |
| Key custody on the host | Total role/legacy key separation and role-specific signature verification | Custody ceremony, host-held producer/actor keys, hardware review keys, external notary signer |
| Revocation | Removed producer keys cannot authenticate new records; inherited history is unchanged | Permission-loss handling, authorized registry transition and finalized-chain activation |

The package reuses the canonicalization/tree implementation from #1511 and
the original lineage primitives from #1662. The contributor work plan is #1661;
#1628 remains open for end-to-end admission. The integrated implementation
targets `main` so it receives the repository's normal PR checks.

## Validation

Use ephemeral fixture keys only. Exercise all pairwise role/legacy aliases;
all cross-scope signatures; unknown, rotated, malformed, and tampered keys;
noncanonical bodies and sidecars; v33 per-draw validation; missing correction
countersignatures; wrong lane/epoch; append-only history; aliases/orphan
sidecars; multifault reasons; and candidate-controlled registry changes.
Git-backed command tests must establish that uncommitted files cannot change
the result and that a successful diagnostic never claims admission.

```bash
uv run pytest tests/notary --no-cov -q
uv run ruff check pyproject.toml src/axiom_encode scripts tests
uv run ruff format --check src/ tests/
uv run python -m compileall -q src/axiom_encode scripts
```
