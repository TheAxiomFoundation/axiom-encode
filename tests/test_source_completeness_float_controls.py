"""A scoped dependency memo must follow native floating-point behavior.

These tests change actual thread-local hardware controls, independently of the
Decimal context and the C rounding direction, and restore the full environment.
"""

from __future__ import annotations

import contextlib
import ctypes
import platform
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

_FTZ = 1 << 15
_DAZ = 1 << 6


class _GlibcX86Environment(ctypes.Structure):
    # glibc sysdeps/x86/fpu/bits/fenv.h declares the x87 environment fields
    # through __glibc_reserved5 first, then __mxcsr on x86_64 (offset 28).
    # https://github.com/bminor/glibc/blob/master/sysdeps/x86/fpu/bits/fenv.h
    _fields_ = [("x87", ctypes.c_char * 28), ("mxcsr", ctypes.c_uint32)]


class _DarwinX86Environment(ctypes.Structure):
    # Read from the macOS SDK's usr/include/fenv.h: __control, __status,
    # __mxcsr, and __reserved[8], making MXCSR's offset 4 and fenv_t's size 16.
    # Its FE_DFL_DISABLE_SSE_DENORMS_ENV sets both DAZ and FTZ; use the
    # documented fenv_t layout here so the two controls can change separately.
    _fields_ = [
        ("control", ctypes.c_ushort),
        ("status", ctypes.c_ushort),
        ("mxcsr", ctypes.c_uint32),
        ("reserved", ctypes.c_char * 8),
    ]


class _HardwareFloatControls:
    def __init__(self, library, environment_type):
        self.library = library
        self.environment_type = environment_type

    @contextlib.contextmanager
    def mode(self, *, ftz=False, daz=False, rounding=0):
        original = self.environment_type()
        assert self.library.fegetenv(ctypes.byref(original)) == 0
        changed = self.environment_type.from_buffer_copy(original)
        changed.mxcsr &= ~(_FTZ | _DAZ)
        changed.mxcsr |= (_FTZ if ftz else 0) | (_DAZ if daz else 0)
        try:
            assert self.library.fesetenv(ctypes.byref(changed)) == 0
            assert self.library.fesetround(rounding) == 0
            observed = self.environment_type()
            assert self.library.fegetenv(ctypes.byref(observed)) == 0
            assert observed.mxcsr & (_FTZ | _DAZ) == changed.mxcsr & (_FTZ | _DAZ)
            assert self.library.fegetround() == rounding
            yield
        finally:
            assert self.library.fesetenv(ctypes.byref(original)) == 0


@pytest.fixture(scope="module")
def hardware_float_controls():
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("Independent FTZ/DAZ controls require x86_64 SSE hardware")
    if sys.platform == "darwin":
        environment_type = _DarwinX86Environment
    elif sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc":
        environment_type = _GlibcX86Environment
    else:
        pytest.skip("fenv_t layout is verified for macOS and glibc x86_64 only")
    try:
        library = ctypes.CDLL(None)
        get_environment = library.fegetenv
        set_environment = library.fesetenv
        get_rounding = library.fegetround
        set_rounding = library.fesetround
    except (AttributeError, OSError, TypeError, RuntimeError):
        pytest.skip("Native floating-point environment API is unavailable")
    for function in (get_environment, set_environment):
        function.argtypes = (ctypes.POINTER(environment_type),)
        function.restype = ctypes.c_int
    get_rounding.argtypes = ()
    get_rounding.restype = ctypes.c_int
    set_rounding.argtypes = (ctypes.c_int,)
    set_rounding.restype = ctypes.c_int
    return _HardwareFloatControls(library, environment_type)


def _denormal_case(control):
    # All case inputs are normal floats, so changing DAZ cannot itself change
    # their repr and accidentally invalidate the memo's input digest.
    product = " * ".join(["a"] * 16)
    if control == "daz":
        # Generate a subnormal intermediate, then scale it back to normal.
        product += " * b"
    rule = {
        "name": "x",
        "kind": "derived",
        "dtype": "Boolean",
        "versions": [{"formula": f"{product} > 0 and local_missing or foreign"}],
        "metadata": {
            "proof": {
                "atoms": [{"import": {"hash": "sha256:imported", "output": "foreign"}}]
            }
        },
    }
    rules = {
        "x": rule,
        "local_missing": {
            "name": "local_missing",
            "kind": "derived",
            "dtype": "Boolean",
            "versions": [{"formula": "omitted_input"}],
        },
    }
    case = {
        "name": "denormals",
        "input": {"a": 1e-20, "b": 1e300},
        "output": {"x": True},
    }
    return rules, case, {}


def _check_denormal_sequence(hardware, control, enabled_states):
    rules, case, environment = _denormal_case(control)
    with sc._case_dependency_environment_memo_scope():
        for enabled in enabled_states:
            with hardware.mode(**{control: enabled}):
                memoized = sc._case_dependency_environment(
                    rules,
                    case,
                    formula_environment=environment,
                    require_asserted_value=True,
                )
                fresh = sc._evaluate_case_dependency_environment(
                    rules,
                    case,
                    formula_environment=environment,
                    require_asserted_value=True,
                )
            # With denormals disabled, only the imported foreign output is
            # reached. Otherwise the omitted local input prevents corroboration.
            assert fresh == ({"x": True} if enabled else {})
            assert memoized == fresh


@pytest.mark.parametrize("control", ("ftz", "daz"))
@pytest.mark.parametrize("enabled_states", ((False, True, False), (True, False, True)))
def test_memo_matches_fresh_across_independent_denormal_control_transitions(
    hardware_float_controls, control, enabled_states
):
    _check_denormal_sequence(hardware_float_controls, control, enabled_states)


@settings(max_examples=40, deadline=None, database=None)
@example(control="ftz", enabled_states=[True, False])
@example(control="daz", enabled_states=[False, True])
@given(
    control=st.sampled_from(("ftz", "daz")),
    enabled_states=st.lists(st.booleans(), min_size=2, max_size=8),
)
def test_memo_matches_fresh_across_hardware_denormal_control_sequences(
    hardware_float_controls, control, enabled_states
):
    _check_denormal_sequence(hardware_float_controls, control, enabled_states)


def test_behavior_probe_distinguishes_independent_denormal_controls(
    hardware_float_controls,
):
    signatures = []
    for ftz, daz in ((False, False), (True, False), (False, True), (True, True)):
        with hardware_float_controls.mode(ftz=ftz, daz=daz):
            signatures.append(sc._floating_point_behavior())
    assert len(set(signatures)) == 4


def test_behavior_probe_distinguishes_rounding_for_both_signs(hardware_float_controls):
    next_positive = float.fromhex("0x1.0000000000001p+0")
    next_negative = float.fromhex("-0x1.0000000000001p+0")
    expected = {
        0: (next_positive, next_negative),  # FE_TONEAREST
        0x400: (1.0, next_negative),  # FE_DOWNWARD
        0x800: (next_positive, -1.0),  # FE_UPWARD
        0xC00: (1.0, -1.0),  # FE_TOWARDZERO
    }
    signatures = []
    for mode, rounded_pair in expected.items():
        with hardware_float_controls.mode(rounding=mode):
            signature = sc._floating_point_behavior()
        assert signature[:16] == struct.pack("!dd", *rounded_pair)
        signatures.append(signature)
    assert len(set(signatures)) == 4


def test_memo_preserves_constant_evaluation_with_native_underflow_unmasked(
    hardware_float_controls,
):
    # A trapping probe can terminate its interpreter before Python can handle
    # an exception. Isolate the regression, disable core dumps, and bound it.
    root = Path(__file__).resolve().parents[1]
    script = r"""
import ctypes
import resource
import sys

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/tests")
from axiom_encode.harness import source_completeness as sc
from test_source_completeness_float_controls import (
    _DarwinX86Environment, _GlibcX86Environment,
)

environment_type = (
    _DarwinX86Environment if sys.platform == "darwin" else _GlibcX86Environment
)
library = ctypes.CDLL(None)
for function in (library.fegetenv, library.fesetenv):
    function.argtypes = (ctypes.POINTER(environment_type),)
    function.restype = ctypes.c_int
original = environment_type()
assert library.fegetenv(ctypes.byref(original)) == 0
changed = environment_type.from_buffer_copy(original)
# Clear the SSE underflow flag and unmask its exception, leaving every other
# exception masked. Native arithmetic that produces a tiny inexact value traps.
changed.mxcsr |= 0x1F80
changed.mxcsr &= ~((1 << 4) | (1 << 11))
rules = {
    "out": {
        "name": "out", "kind": "derived", "dtype": "Integer",
        "versions": [{"formula": "1"}],
    }
}
case = {"name": "constant", "input": {}, "output": {}}
environment = {}
kwargs = {"formula_environment": environment, "require_asserted_value": False}
try:
    assert library.fesetenv(ctypes.byref(changed)) == 0
    observed = environment_type()
    assert library.fegetenv(ctypes.byref(observed)) == 0
    if observed.mxcsr & (1 << 11):
        print("Host keeps native underflow exceptions masked")
        raise SystemExit(77)
    assert sc._floating_point_behavior() is None
    fresh = sc._evaluate_case_dependency_environment(rules, case, **kwargs)
    with sc._case_dependency_environment_memo_scope():
        memoized = sc._case_dependency_environment(rules, case, **kwargs)
finally:
    assert library.fesetenv(ctypes.byref(original)) == 0
assert fresh == memoized == {"out": 1}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, (
        f"Child returned {completed.returncode}\n" + completed.stdout + completed.stderr
    )
