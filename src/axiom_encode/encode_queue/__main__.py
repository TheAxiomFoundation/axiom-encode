"""``python -m axiom_encode.encode_queue`` runs ``axiom-encode queue``."""

import sys

from axiom_encode.encode_queue.cli import run_queue

if __name__ == "__main__":
    raise SystemExit(run_queue(sys.argv[1:]))
