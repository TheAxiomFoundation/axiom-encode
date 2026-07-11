Add the separately built Go signing supervisor and broker hard cut for apply,
retire, and evaluation Ed25519 operations. External signer protocol v2 and
broker protocol v4 domain-bind persisted signatures to distinct apply/eval
scopes. Every invocation reads pairwise-distinct apply, eval, and corpus-release
roots from one protected config and rejects environment trust inputs. Python
starts through a protected direct `-I -S`
bootstrap inside one self-contained root-owned runtime, validates its complete
import/origin closure before attaching the peer-authenticated broker, and
forwards only a constructed minimal environment. Apply manifests, eval
verdicts, and homogeneous eval suite artifacts move to v5 without legacy
translation. Production rejects caller-writable/injected runtimes, elevated
UID/capability state, file capabilities, ambiguous protocol frames, and
inheritable signing capabilities.
