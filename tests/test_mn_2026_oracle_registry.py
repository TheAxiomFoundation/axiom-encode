import hashlib
import re
import tomllib
from pathlib import Path

import yaml
from axiom_oracles.bridges.coverage import build_policyengine_coverage_report
from axiom_oracles.bridges.registry import load_policyengine_registry

MODULE = "us-mn:policies/income_tax/pilot_liability_pipeline"
OUTPUT = "mn_pit_pilot_schedule_tax"
POLICYENGINE_VARIABLE = "mn_basic_tax"
ORACLE_MERGE = "9434ce833bacbc60bc1572ea0245913ff4e48501"
ENCODER_VERSION = "0.2.2297"
