# Canadian reviewed artifact head 94789006

This record supports the exact Canadian head admitted by encoder PR [#1984](https://github.com/TheAxiomFoundation/axiom-encode/pull/1984). The [code-owner review](https://github.com/TheAxiomFoundation/axiom-encode/pull/1984#pullrequestreview-5623605052) checked the matching and publication invariants; it requested a new version and auditable provenance. This record does not claim that review has approved the revised head.

## Immutable scope

- Consumer PR: [rulespec-ca #28](https://github.com/TheAxiomFoundation/rulespec-ca/pull/28).
- Admitted commit: [`94789006ce372a1967c258ef47c12b15a1fe511e`](https://github.com/TheAxiomFoundation/rulespec-ca/commit/94789006ce372a1967c258ef47c12b15a1fe511e); tree `572adba8d24fc875218b510da1d637717164f6e4`.
- [Diff from parent `ce8e35c36a168449159d242f98d97c61a6390521`](https://github.com/TheAxiomFoundation/rulespec-ca/compare/ce8e35c36a168449159d242f98d97c61a6390521...94789006ce372a1967c258ef47c12b15a1fe511e): exactly three added T1 Step 2 files. All 529 files tracked at the parent retain identical blob bytes. That is a preservation check, **not** a fresh semantic audit of every prior file.
- Schedule M and L and their receipts already exist at this head; they are not additions in that three-file diff.
- Step 2's artifact-only generation is [encoder run 39536963804](https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/39536963804), producer `882541f454fcc9057695fb67e397878bb65118e0` / `0.2.2242`, engine `89571cc2a938707fd60a5489b951135345697296`.

## Admission semantics

The exact `(ca, full commit SHA)` pair is admitted for **all supported protected artifact-only modes**, including manifest refresh, reviewed-candidate promotion, model encoding, canonical refresh and supported corpus-artifact transport. The entry is not a refresh-only restriction. Every mode retains its own validation, signature, budget and protected-environment checks. The entry does not add a `REVIEWED_RULESPEC_PR_BASES` tuple, authorize PR publication, or itself dispatch or sign anything.

The immediate intended use is byte-preserving M/L manifest refresh, followed by separately reviewed T1 work. This intent does not narrow the allowlist's actual mode coverage.

## Consumer pin and validation limits

The [committed workflow toolchain](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/.axiom/workflow-toolchain.toml) is **0.2.2230 / `15aa35772aa186c2b663d544c2b3bc400ebc164a`**, not 2242. Consumer 2242 alignment is a proposed later change, conditional on compatible M/L receipts and fresh validation. It is not included here.

Historical local diagnostics found M/L ordinary source validation passing with 2243 and Step 2 with 2242. Those are local diagnostic results, not hosted consumer CI clearance or full policy readiness. In particular, the older consumer parser and M/L receipt compatibility remain separate publication prerequisites. The native Step 2 run above is publicly addressable evidence; the local diagnostics are not substituted for a hosted run.

## Exact tracked artifacts

These SHA-256 digests identify the reviewed bytes. A digest alone does not prove a signature; the supported manifest verifier must authenticate the receipt and all bindings again before any protected operation.

| Artifact at the admitted commit | SHA-256 |
| --- | --- |
| [.axiom/encoding-manifests/ca/policies/cra/t1-2025/t1-income-net-taxable-income/step-2.json](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/.axiom/encoding-manifests/ca/policies/cra/t1-2025/t1-income-net-taxable-income/step-2.json) | `d1aacbf46e42cdb868b110a0d150393fe49cc4e2658e38a3fe246ff1ec841ca6` |
| [.axiom/encoding-manifests/ca/policies/revenu-quebec/tp1-2025/schedule-l-business-income.json](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/.axiom/encoding-manifests/ca/policies/revenu-quebec/tp1-2025/schedule-l-business-income.json) | `c71a980631611b0ddac70e88c996387afa279bebec9a81b1571d668d023fbab2` |
| [.axiom/encoding-manifests/ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.json](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/.axiom/encoding-manifests/ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.json) | `377a76126b82eabd7241db5715b3ac879bf1534d32c7ecdc37c1c0fdee0ed9b8` |
| [ca/policies/cra/t1-2025/t1-income-net-taxable-income/step-2.test.yaml](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/ca/policies/cra/t1-2025/t1-income-net-taxable-income/step-2.test.yaml) | `648fcd7703eed0764ffba9c451dd47797702c1b94e2afaea2deca721c2607250` |
| [ca/policies/cra/t1-2025/t1-income-net-taxable-income/step-2.yaml](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/ca/policies/cra/t1-2025/t1-income-net-taxable-income/step-2.yaml) | `8a6775eee53c35028680361004bd645a5e4efe9dc0daa170ed8deb646f5f111a` |
| [ca/policies/revenu-quebec/tp1-2025/schedule-l-business-income.test.yaml](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/ca/policies/revenu-quebec/tp1-2025/schedule-l-business-income.test.yaml) | `43f8bb1d8ea3d0ee7926f7fb487b191b300e27e5dd3ef771753376cea7168c09` |
| [ca/policies/revenu-quebec/tp1-2025/schedule-l-business-income.yaml](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/ca/policies/revenu-quebec/tp1-2025/schedule-l-business-income.yaml) | `04f2b4380addb560dd1e19bcce4e2aad31a6b8783617f9e0245fe8cd47e4ca5b` |
| [ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.test.yaml](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.test.yaml) | `91fd2eadbfff8d52abb7fc5390926264002fda1ad1f966c0d3c3efe314101e0b` |
| [ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.yaml](https://github.com/TheAxiomFoundation/rulespec-ca/blob/94789006ce372a1967c258ef47c12b15a1fe511e/ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.yaml) | `18cb6474a21ba1333bda29c1b04c1bfc30a0f125d34d3a464065f0e180563eee` |

## Reproduce the byte and base checks

In an official `rulespec-ca` clone with both commits fetched, `git diff --name-status ce8e35c36a168449159d242f98d97c61a6390521 94789006ce372a1967c258ef47c12b15a1fe511e` must show only the three Step 2 additions above. `git ls-tree -r` at the parent enumerates the 529 preserved paths; compare their blob IDs at both commits. Hash each linked file's raw bytes to reproduce the table.

The revised admission was also exercised against a clean, real checkout at the exact head (no mocked Git): `validate_rulespec_base(repo, "ca", head, open_pr=False)` returned `reviewed-head-artifact`; `open_pr=True` refused the unapproved publication base, and country `us` refused the identity. This is a base-admission probe only, not a signer invocation, manifest verification or source-validation result. The repository tests additionally check removal, identity and publication restrictions.
