import hashlib
import re
import tomllib
from pathlib import Path

import yaml
from axiom_oracles.bridges.coverage import build_policyengine_coverage_report
from axiom_oracles.bridges.registry import load_policyengine_registry

MODULE = "us-sc:policies/income_tax/pilot_liability_pipeline"
OUTPUT_NAME = "sc_pit_pilot_income_tax_liability"
INPUT_NAME = "sc_pit_pilot_state_taxable_income"
POLICYENGINE_VARIABLE = "sc_income_tax_before_non_refundable_credits"
ORACLE_MERGE = "9434ce833bacbc60bc1572ea0245913ff4e48501"
ENCODER_VERSION = "0.2.2297"
