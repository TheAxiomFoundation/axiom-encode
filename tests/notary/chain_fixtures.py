"""An entirely ephemeral signed epoch, activation and local-output receipt."""

from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
from axiom_encode.notary.chain import BOOTSTRAP_PATHS, Anchor, reconstruct
from axiom_encode.notary.lineage import STORE_PREFIX
from axiom_encode.notary.manifest import manifest_diff, manifest_sha256
from axiom_encode.notary.verification import Snapshot, verify_snapshots

from .lineage_fixtures import LANE, Identities, generation, policy_body, public_entry
from .test_protocol import candidate, profile
from .test_verification import snapshot


def addressed(body):
    raw = jcs_dumps(body)
    return sha256_hex(raw), raw


@dataclass
class Epoch:
    identities: Identities
    anchor: Anchor
    history: list[Snapshot]
    base: Snapshot
    active: Snapshot
    inventory: bytes

    def append(self, files, *, head=None):
        blobs = dict(self.history[-1].blobs) if self.history else {}
        blobs.update(files)
        if head:
            blobs["HEAD.json"] = jcs_dumps(
                {
                    "schema": "axiom/notary-head/v1",
                    "tip_sha256": head[0],
                    "tip_kind": head[1],
                }
            )
        self.history.append(snapshot(f"{len(self.history) + 1:040x}", blobs))

    def state(self):
        return reconstruct(self.history, self.anchor)

    def signed(self, body, roles):
        address, raw = addressed(body)
        return address, {address + ".json": raw} | {
            address + f".json.{role}.sig": self.identities.sidecar(raw, role)
            for role in roles
        }

    def finalize(self, address, kind, manifest, sequence, *, voids=()):
        marker = {
            "schema": "axiom/notary-finalization/v1",
            "lane": LANE,
            "epoch_sha256": self.anchor.epoch_sha256,
            "target_sha256": address,
            "target_kind": kind,
            "merged_tip_manifest_sha256": manifest,
            "sequence": str(sequence),
        }
        digest, raw = addressed(marker)
        files = {digest + ".json": raw}
        for target, target_kind in voids:
            void = {
                "schema": "axiom/notary-void/v1",
                "lane": LANE,
                "epoch_sha256": self.anchor.epoch_sha256,
                "target_sha256": target,
                "target_kind": target_kind,
                "reason": "superseded",
            }
            v, raw = addressed(void)
            files[v + ".json"] = raw
        self.append(files, head=(address, kind))

    @classmethod
    def create(cls):
        identities = Identities.create()
        policy = policy_body()
        transition_policy = {
            "schema": "axiom/notary-transition-path-policy/v1",
            "lane": LANE,
            "rules": [
                {"action": "include", "prefix": ".axiom/notary"},
                {"action": "include", "prefix": ".github"},
            ],
        }
        prospective = dict(
            zip(
                BOOTSTRAP_PATHS.values(),
                [
                    jcs_dumps(policy),
                    jcs_dumps(transition_policy),
                    jcs_dumps(profile()),
                    jcs_dumps(identities.body),
                ],
                strict=True,
            )
        )
        consumer = {
            "schema": "axiom/notary-consumer/v1",
            "lane": LANE,
            "epoch_sha256": "0" * 64,
            "notary_repository": LANE + "-notary",
            "notary_spki_sha256": identities.pins["notary_spki_sha256"],
        }
        waiver = b"validate_failures: {}\n"
        base = snapshot(
            "a" * 40,
            {
                "known-validation-gaps.yaml": waiver,
                ".axiom/toolchain.toml": (
                    f'[toolchain]\naxiom_corpus_release="fixture"\naxiom_corpus_release_content_sha256="{"c" * 64}"\nvalidation_waiver_set_sha256="{sha256_hex(waiver)}"\n'
                ).encode(),
            },
        )
        roots = {}
        for name in ("legacy_apply_root", "legacy_eval_root"):
            public = identities.keys[name].public_key()
            roots[name] = {
                "raw_key_id": "sha256:"
                + sha256_hex(
                    public.public_bytes(
                        serialization.Encoding.Raw, serialization.PublicFormat.Raw
                    )
                ),
                "public_key_spki_der_base64": public_entry(identities.keys[name])[
                    "public_key_spki_der_base64"
                ],
            }
        body = {
            "schema": "axiom/notary-genesis/v1",
            "lane": LANE,
            "genesis_commit_git_oid": base.commit,
            "genesis_tree_manifest_sha256": manifest_sha256(base.manifest),
            "bootstrap_policies": {
                name: sha256_hex(prospective[path])
                for name, path in BOOTSTRAP_PATHS.items()
            },
            "activation_spec_template_sha256": sha256_hex(jcs_dumps(consumer)),
            "consumer_spec_path": ".axiom/notary/consumer.json",
            "notary_repository": LANE + "-notary",
            **roots,
            "v5_attested": [],
            "baseline_unattested": [],
        }
        address, _ = addressed(body)
        anchor = Anchor(
            LANE, address, LANE + "-notary", identities.pins["notary_spki_sha256"]
        )
        consumer["epoch_sha256"] = address
        active = snapshot(
            "b" * 40,
            base.blobs
            | prospective
            | {body["consumer_spec_path"]: jcs_dumps(consumer)},
        )
        inventory = jcs_dumps(
            {
                "schema": "axiom/notary-dependency-inventory/v1",
                "lane": LANE,
                "actions": [],
                "containers": [],
                "python_lock_sha256": "d" * 64,
                "verifier": {
                    "repo": "TheAxiomFoundation/axiom-encode",
                    "git_oid": "e" * 40,
                },
            }
        )
        epoch = cls(identities, anchor, [], base, active, inventory)
        _, bundle = epoch.signed(body, ("genesis", "admin-approver"))
        bundle.update({sha256_hex(raw) + ".raw": raw for raw in prospective.values()})
        epoch.append(bundle)
        epoch.finalize(address, "genesis", body["genesis_tree_manifest_sha256"], 1)
        changes = manifest_diff(base.manifest, active.manifest)
        transition = {
            "schema": "axiom/notary-transition/v1",
            "lane": LANE,
            "epoch_sha256": address,
            "chain_predecessor_sha256": address,
            "chain_predecessor_kind": "genesis",
            "base_tree_manifest_sha256": manifest_sha256(base.manifest),
            "subject_tree_manifest_sha256": manifest_sha256(active.manifest),
            "subject_commit_git_oid": active.commit,
            "delta": [c._asdict() for c in changes],
            "reason": "activate fixture epoch",
        }
        t, bundle = epoch.signed(transition, ("transition", "admin-approver"))
        bundle.update(
            {c.after_entry_sha256 + ".raw": active.blobs[c.path] for c in changes}
        )
        epoch.append(bundle)
        epoch.finalize(t, "transition", manifest_sha256(active.manifest), 2)
        return epoch

    def receipt(self, *, label="draw", run_id="1"):
        body = generation()
        body["epoch_sha256"], body["draw_set_id"] = self.anchor.epoch_sha256, label
        subject = snapshot(
            "c" * 40,
            self.active.blobs
            | {"rules/example.yaml": b"generated\n"}
            | {
                STORE_PREFIX + name: file.raw
                for name, file in self.identities.store(body).items()
            },
        )
        raw_report = verify_snapshots(
            self.active,
            subject,
            self.state().predecessor(),
            self.inventory,
            [{"gate_id": "compile", "outcome": "pass"}],
        )
        report = strict_parse(raw_report)
        cbody = report | {
            "schema": "axiom/notary-receipt-candidate/v1",
            "report_sha256": sha256_hex(raw_report),
            "job1": candidate()["job1"] | {"run_id": run_id},
        }
        c, bundle = self.signed(cbody, ("approver",))
        bundle[cbody["report_sha256"] + ".json"] = raw_report
        rbody = {
            "schema": "axiom/notary-receipt/v1",
            "lane": LANE,
            "epoch_sha256": self.anchor.epoch_sha256,
            "candidate_sha256": c,
            "authorization": {
                "environment": "notary-signing",
                "approve_check_run_id": "4",
                "approval_signature_sha256": sha256_hex(
                    bundle[c + ".json.approver.sig"]
                ),
            },
        }
        r, signed = self.signed(rbody, ("notary",))
        return r, bundle | signed, subject
