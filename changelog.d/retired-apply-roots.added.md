Preserve owner-frozen historical apply manifests and receipts after root rotation
with optional retired verification roots and an immutable repository/path/byte
inventory. Persisted readers and explicit retrospective guards admit only frozen
artifacts and their historical producer identities; signing, ordinary change
guards, and the signer challenge retain the active root. Reject changed,
relocated, newly introduced, or restored live old-key artifacts after
supersession, and require exact frozen evidence for nested historical signatures.
