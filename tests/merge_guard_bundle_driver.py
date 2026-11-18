"""Drive the merge guard bundle under ``python -I -S``. A test helper.

``tests/test_merge_guard_enforce_workflow.py`` runs this file in a subprocess::

    python -I -S merge_guard_bundle_driver.py sweep <bundle> <scenario.json>
    python -I -S merge_guard_bundle_driver.py foreign <bundle> <directory>

It is not part of the bundle and the workflow never runs it. It imports the
standard library only, loads the bundle's ``launch.py`` by path, and prints
one JSON object.

``sweep`` calls the launcher's ``prepare()``, then runs
``merge_guard_runner.enforce_command`` once per case of the scenario, each
over a fake HTTP opener that serves the case's recorded responses, and
reports the exit codes, the reports and ``sorted(sys.modules)``.

``foreign`` puts a directory that holds another ``axiom_encode`` package
first on ``sys.path`` and reports what the launcher's ``main()`` returns.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys

ORIGIN = "https://api.github.com"
# Loading ``launch.py`` by path would otherwise write its bytecode into the
# bundle directory, which the launcher requires to hold exactly the bundle.
sys.dont_write_bytecode = True


def load_launcher(bundle):
    spec = importlib.util.spec_from_file_location(
        "merge_guard_launch_under_test", os.path.join(bundle, "launch.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, raw):
        self.status = 200
        self._raw = raw

    def read(self, limit):
        return self._raw[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class Opener:
    """Serves ``api``: a path maps to its responses, in order; the last repeats."""

    def __init__(self, api):
        self.api = api
        self.seen = {}
        self.methods = []
        self.bodies = 0

    def open(self, request, timeout):
        self.methods.append(request.get_method())
        self.bodies += request.data is not None
        if not request.full_url.startswith(ORIGIN + "/repos/"):
            raise AssertionError("request outside the fixed origin")
        path = request.full_url[len(ORIGIN) :]
        responses = self.api[path]
        index = min(self.seen.get(path, 0), len(responses) - 1)
        self.seen[path] = self.seen.get(path, 0) + 1
        return Response(json.dumps(responses[index]).encode())


class Clock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def sweep(bundle, scenario_path):
    launch = load_launcher(bundle)
    prepared = launch.prepare()
    from axiom_encode import merge_guard_runner as runner

    with open(scenario_path, encoding="utf-8") as stream:
        scenario = json.load(stream)
    results = []
    for case in scenario["cases"]:
        opener, clock, out = Opener(case["api"]), Clock(), io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runner.enforce_command(
                environ=case["environ"],
                opener=opener,
                sleep=clock.sleep,
                clock=clock.clock,
            )
        results.append(
            {
                "name": case["name"],
                "code": code,
                "report": json.loads(out.getvalue()),
                "methods": sorted(set(opener.methods)),
                "bodies": opener.bodies,
            }
        )
    # A module that is in the bundle directory but not on the list. Written
    # after prepare(), which refuses a directory that holds it.
    unlisted = os.path.join(bundle, "axiom_encode", "unlisted.py")
    with open(unlisted, "w", encoding="utf-8") as stream:
        stream.write("VALUE = 1\n")
    refusals = {}
    for name in ("axiom_encode.unlisted", "axiom_encode.cli", "axiom_encode.harness"):
        try:
            importlib.import_module(name)
            refusals[name] = None
        except ImportError as exc:
            refusals[name] = str(exc)
    os.remove(unlisted)
    return {
        "prepared": prepared,
        "results": results,
        "modules": sorted(sys.modules),
        "refusals": refusals,
        "loaded_outside": launch.loaded_outside(bundle),
        "bundle_is_last_on_path": sys.path[-1] == bundle,
        "pycache": os.path.exists(os.path.join(bundle, "axiom_encode", "__pycache__")),
        "flags": [sys.flags.isolated, sys.flags.no_site],
    }


def foreign(bundle, directory):
    launch = load_launcher(bundle)
    sys.path.insert(0, directory)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = launch.main()
    return {"code": code, "output": out.getvalue()}


if __name__ == "__main__":
    mode, bundle = sys.argv[1], os.path.abspath(sys.argv[2])
    if mode == "sweep":
        print(json.dumps(sweep(bundle, sys.argv[3])))
    elif mode == "foreign":
        print(json.dumps(foreign(bundle, sys.argv[3])))
    else:
        sys.exit(2)
