"""Reference adapter for Pavel's review: B1.6 note 50/52 only.

Bundle this file with the separately reviewed generate_incidence_tables.py.
Both files must be measured in enrollment. Stage the captured, release-bound
notes artifact as inputs/notes.jsonl; the generator also checks its own hash.
"""

import importlib.util
import json
import sys
from pathlib import Path

PREFIX = "us/policies/usitc/us-tariff-incidence/generated/"
OUTPUTS = sorted(
    PREFIX + name + extension
    for name in ("note50-brazil-exemptions", "note52-reciprocal-exemptions")
    for extension in (".yaml", ".test.yaml")
)


def main():
    request = json.loads(Path(sys.argv[1]).read_bytes())
    if (
        set(request) != {"schema", "input_root", "output_root", "parameters", "outputs"}
        or request["schema"] != "axiom/deterministic-adapter-request/v1"
        or request["parameters"] != {"actions": "brazil-50,reciprocal-52"}
        or request["outputs"] != OUTPUTS
    ):
        raise SystemExit("B1.6 adapter requires the exact note 50/52 recipe")
    spec = importlib.util.spec_from_file_location(
        "enrolled_b16_generator",
        Path(__file__).with_name("generate_incidence_tables.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    destination = Path(request["output_root"]) / PREFIX
    result = module.generate(
        destination,
        Path(request["input_root"]) / "notes.jsonl",
        {"brazil-50", "reciprocal-52"},
    )
    if sorted(PREFIX + name for name in result) != OUTPUTS:
        raise SystemExit("B1.6 generator returned an unexpected output set")


if __name__ == "__main__":
    main()
