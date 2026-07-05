"""Thin re-export shim: this module moved to :mod:`axiom_oracles.bridges.ecps_tax`.

The PolicyEngine/Populace oracle-bridge layer was extracted to the
axiom-oracles repo (axiom_oracles/bridges/README.md there documents the
public API; TheAxiomFoundation/axiom-oracles#124). Replacing this module in
``sys.modules`` with the bridge module keeps every existing import path,
attribute access, and monkeypatch target — and therefore the CLI — behaving
exactly as before the move. Do not add code here: edit the bridge module in
axiom-oracles instead. Deleting these shims (and the now-inert sibling data
directories) is deferred until consumers import axiom_oracles.bridges
directly.
"""

import sys

import axiom_oracles.bridges.ecps_tax as _bridge

sys.modules[__name__] = _bridge
