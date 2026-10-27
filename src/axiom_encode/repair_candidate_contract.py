"""Shared bounds for validator-rejected repair candidates."""

VALIDATION_RETRY_CANDIDATE_MAX_FILE_BYTES = 512 * 1024
VALIDATION_RETRY_CANDIDATE_MAX_TOTAL_BYTES = 1024 * 1024

# Diagnostic prose is larger than executable YAML; keep its bound independent.
FAILED_ENCODE_CANDIDATE_MAX_ISSUES_BYTES = 8 * 1024 * 1024
