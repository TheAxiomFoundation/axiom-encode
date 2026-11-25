"""The replacement dry-run check: interface, program bindings, declared change."""

import hashlib
import json

import pytest
import yaml

import axiom_encode.replacement_check as replacement_check
from axiom_encode.replacement_check import (
    NOT_CHECKED,
    ReplacementCheckError,
    check_replacement,
    first_difference,
    main,
)

OLD = "us-az/policies/des/ece/expanded.yaml"


def rule(name, formula, kind="derived", entity="Household", versions=None, **extra):
    body = {"name": name, "kind": kind, "dtype": "Judgment", "period": "Month"}
    if kind == "parameter":
        body = {"name": name, "kind": kind, "dtype": "Rate"}
    else:
        body["entity"] = entity
    body.update(extra)
    body["versions"] = versions or [
        {"effective_from": "0001-01-01", "formula": formula}
    ]
    return body


def module(*rules, imports=None, inputs=None, **module_fields):
    payload = {"format": "rulespec/v1", "module": {"summary": "x", **module_fields}}
    if imports:
        payload["imports"] = imports
    payload["rules"] = list(rules)
    if inputs:
        payload["inputs"] = inputs
    return payload


def old_rules():
    return [
        rule("limit_rate", "1.85", kind="parameter"),
        rule("within_limit", "gross_income <= limit_rate * fpl_amount"),
        rule("unread_flag", "within_limit"),
        rule("ece", "within_limit and not disqualified"),
    ]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "rulespec"
    write(root / OLD, module(*old_rules()))
    # A same-jurisdiction #name importer and a whole-module importer.
    write(
        root / "us-az/policies/des/ece/benefit.yaml",
        module(
            rule("blocked", "not limit_rate"),
            imports=["policies/des/ece/expanded#limit_rate"],
        ),
    )
    write(
        root / "us-az/policies/des/other/user.yaml",
        module(
            rule("uses_ece", "ece and something_else"),
            imports=["us-az:policies/des/ece/expanded"],
        ),
    )
    # Not an importer: names the path in text only.
    write(
        root / "us-az/policies/des/other/mention.yaml",
        module(rule("mention", "policies_des_ece_expanded")),
    )
    write(
        root / "us-az/policies/des/ece/member.yaml",
        module(rule("member_rule", "within_limit or x")),
    )
    write(
        root / "programs/us-az/snap/fy-2026.yaml",
        {
            "program": "us-az/snap",
            "outputs": ["snap_eligible"],
            "scope": {
                "federal": ["regulations/7-cfr/273/9"],
                "state": ["policies/des/ece/expanded", "policies/des/ece/member"],
            },
            "transformations": [
                {
                    "pattern": "sum_terms",
                    "name": "gross_income",
                    "terms": ["snap_total_gross_income"],
                },
                {
                    "pattern": "all_of",
                    "name": "snap_eligible",
                    "conditions": ["ece", "residency"],
                },
            ],
        },
    )
    # Every scope entry of every program spec must resolve to a file.
    write(root / "us/regulations/7-cfr/273/9.yaml", module(rule("federal_rule", "1")))
    write(root / "us-az/policies/des/tanf/x.yaml", module(rule("tanf_rule", "1")))
    write(
        root / "programs/us-az/tanf/fy-2026.yaml",
        {"program": "us-az/tanf", "scope": {"state": ["policies/des/tanf/x"]}},
    )
    return root


def record(tmp_path, declared=(), removed=(), **override):
    payload = {
        "schema": "axiom-encode/replacement-behavior-change/v1",
        "replaced": OLD,
        "source_change": {"citation": "us-az/manual/des/archived-policy/notice"},
        "changes": [
            {"output": name, "effective_from": date, "reason": "185% to 200%"}
            for name, date in declared
        ],
        "removed_exports": [{"name": name, "reason": "unused"} for name in removed],
    }
    payload.update(override)
    return write(tmp_path / "record.yaml", payload)


def candidate(tmp_path, rules, name="new.yaml", imports=None, **fields):
    return write(
        tmp_path / "candidates" / name, module(*rules, imports=imports, **fields)
    )


def raised_rate():
    return rule(
        "limit_rate",
        None,
        kind="parameter",
        versions=[
            {"effective_from": "0001-01-01", "formula": "1.85"},
            {"effective_from": "2026-03-01", "formula": "2.0"},
        ],
    )


def codes(report):
    return sorted(issue["code"] for issue in report["issues"])


def test_dated_raise_with_record_passes_and_reports_readers(repo, tmp_path):
    rules = [raised_rate(), *old_rules()[1:]]
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, [("limit_rate", "2026-03-01")]),
    )
    assert report["passed"] is True, report["issues"]
    assert report["authorizes_signing"] is False
    assert report["form"] == "same_path"
    assert report["behavior_changes"] == [
        {
            "name": "limit_rate",
            "first_difference": "2026-03-01",
            "declared_effective_from": "2026-03-01",
        }
    ]
    assert report["importers"] == [
        "us-az/policies/des/ece/benefit.yaml",
        "us-az/policies/des/other/user.yaml",
    ]
    assert report["programs"] == ["programs/us-az/snap/fy-2026.yaml"]
    assert report["readers"] == {
        "ece": [
            "programs/us-az/snap/fy-2026.yaml",
            "us-az/policies/des/other/user.yaml",
        ],
        "limit_rate": ["us-az/policies/des/ece/benefit.yaml"],
        "within_limit": [
            "us-az/policies/des/ece/member.yaml (via programs/us-az/snap/fy-2026.yaml)"
        ],
    }
    assert report["program_bound_inputs"] == {
        "programs/us-az/snap/fy-2026.yaml": ["gross_income"]
    }
    assert len(report["old"]["sha256"]) == 64


def test_undeclared_change_is_an_issue(repo, tmp_path):
    rules = [raised_rate(), *old_rules()[1:]]
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})
    assert codes(report) == ["undeclared_change"]
    assert report["passed"] is False


def test_new_value_for_all_dates_does_not_keep_old_value(repo, tmp_path):
    rules = [rule("limit_rate", "2.0", kind="parameter"), *old_rules()[1:]]
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, [("limit_rate", "2026-03-01")]),
    )
    assert codes(report) == ["old_value_not_kept"]
    assert "0001-01-01" in report["issues"][0]["message"]


def test_declared_change_must_exist(repo, tmp_path):
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, old_rules())},
        record(
            tmp_path, [("limit_rate", "2026-03-01"), ("no_such_rule", "2026-03-01")]
        ),
    )
    assert codes(report) == ["declared_change_absent", "declared_change_unknown"]


@pytest.mark.parametrize("spelling", [1.850, "1.850", " 1.85 "])
def test_equivalent_numeric_spellings_are_not_a_change(repo, tmp_path, spelling):
    rules = [
        rule("limit_rate", spelling, kind="parameter"),
        rule("within_limit", "gross_income  <=  limit_rate * fpl_amount"),
        *old_rules()[2:],
    ]
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})
    assert report["passed"] is True, report["issues"]
    assert report["behavior_changes"] == []


@pytest.mark.parametrize(
    "removed, reader",
    [
        ("limit_rate", "us-az/policies/des/ece/benefit.yaml"),
        ("ece", "programs/us-az/snap/fy-2026.yaml"),
        ("within_limit", "member.yaml (via programs/us-az/snap/fy-2026.yaml)"),
    ],
)
def test_removing_a_read_export_is_refused_even_when_declared(
    repo, tmp_path, removed, reader
):
    rules = [item for item in old_rules() if item["name"] != removed]
    if removed == "within_limit":
        rules = [
            rule("ece", "gross_income <= limit_rate * fpl_amount and not disqualified")
            if item["name"] == "ece"
            else rule("unread_flag", "ece")
            if item["name"] == "unread_flag"
            else item
            for item in rules
        ]
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, removed=[removed]) if removed != "within_limit" else None,
    )
    assert "export_removed" in codes(report)
    message = next(i for i in report["issues"] if i["code"] == "export_removed")
    assert reader in message["message"]


def test_whole_module_importer_reads_by_formula(repo, tmp_path):
    rules = [item for item in old_rules() if item["name"] != "ece"]
    report = check_replacement(
        repo, OLD, {OLD: candidate(tmp_path, rules)}, record(tmp_path, removed=["ece"])
    )
    assert "us-az/policies/des/other/user.yaml" in report["readers"]["ece"]
    assert "export_removed" in codes(report)


def test_unread_removal_needs_a_declaration(repo, tmp_path):
    rules = [item for item in old_rules() if item["name"] != "unread_flag"]
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})
    assert codes(report) == ["undeclared_removal"]
    declared = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, removed=["unread_flag"]),
    )
    assert declared["passed"] is True, declared["issues"]
    assert declared["exports"]["removed"] == ["unread_flag"]


def test_declared_removal_must_happen(repo, tmp_path):
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, old_rules())},
        record(tmp_path, removed=["unread_flag"]),
    )
    assert codes(report) == ["declared_removal_absent"]


@pytest.mark.parametrize(
    "name, field, value",
    [
        ("limit_rate", "dtype", "Money"),
        ("ece", "entity", "Person"),
        ("within_limit", "period", "Year"),
        ("ece", "unit", "USD"),
        ("ece", "visibility", "private"),
        ("unread_flag", "entity", "Person"),
    ],
)
def test_interface_change(repo, tmp_path, name, field, value):
    rules = [
        dict(item, **{field: value}) if item["name"] == name else item
        for item in old_rules()
    ]
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})
    assert report["exports"]["type_changed"][0]["fields"] == [field]
    assert codes(report) == [
        "export_type_changed"
        if name != "unread_flag"
        else "undeclared_interface_change"
    ]
    assert report["passed"] is False


def test_declared_interface_change_of_an_unread_rule(repo, tmp_path):
    rules = [
        dict(item, entity="Person") if item["name"] == "unread_flag" else item
        for item in old_rules()
    ]
    changes = [{"name": "unread_flag", "reason": "budgetary unit is a Household"}]
    path = record(tmp_path, interface_changes=changes)
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)}, path)
    assert report["passed"] is True, report["issues"]
    same = check_replacement(repo, OLD, {OLD: candidate(tmp_path, old_rules())}, path)
    assert codes(same) == ["declared_interface_change_absent"]


def test_dropped_program_binding(repo, tmp_path):
    rules = [
        old_rules()[0],
        rule("within_limit", "household_gross <= limit_rate * fpl_amount"),
        *old_rules()[2:],
    ]
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, [("within_limit", "0001-01-01")]),
    )
    assert codes(report) == ["program_binding_dropped"]
    assert "gross_income" in report["issues"][0]["message"]


def test_split_into_block_modules_lists_mechanical_updates(repo, tmp_path):
    block4 = "us-az/manual/des/faa5/ece/block-4.yaml"
    block5 = "us-az/manual/des/faa5/ece/block-5.yaml"
    new = {
        block4: candidate(tmp_path, old_rules()[:2], "b4.yaml"),
        # An import of the other new file of the split is not a module change.
        block5: candidate(
            tmp_path,
            old_rules()[2:],
            "b5.yaml",
            imports=["manual/des/faa5/ece/block-4#within_limit"],
        ),
    }
    report = check_replacement(repo, OLD, new)
    assert report["form"] == "split_or_move"
    assert report["module_changes"] == []
    assert codes(report) == ["importers_not_repointed", "program_scope_not_updated"]
    assert report["mechanical_updates"] == [
        {"kind": "repoint_import", "path": "us-az/policies/des/ece/benefit.yaml"},
        {"kind": "repoint_import", "path": "us-az/policies/des/other/user.yaml"},
        {"kind": "program_scope", "path": "programs/us-az/snap/fy-2026.yaml"},
    ]
    assert [item["path"] for item in report["new"]] == [block4, block5]


def test_duplicate_rule_across_new_modules(repo, tmp_path):
    new = {
        "us-az/manual/a/block-1.yaml": candidate(tmp_path, old_rules(), "a.yaml"),
        "us-az/manual/a/block-2.yaml": candidate(tmp_path, old_rules()[:1], "b.yaml"),
    }
    assert "duplicate_new_rule" in codes(check_replacement(repo, OLD, new))


def test_record_fields_are_required(repo, tmp_path):
    path = record(
        tmp_path,
        replaced="us-az/other.yaml",
        source_change={},
        changes=[{"output": "limit_rate", "effective_from": "2026-03-01"}],
    )
    rules = [raised_rate(), *old_rules()[1:]]
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)}, path)
    assert codes(report) == [
        "record_missing_reason",
        "record_missing_source",
        "record_wrong_module",
    ]


def test_record_schema_and_dates_are_strict(repo, tmp_path):
    with pytest.raises(ReplacementCheckError, match="schema"):
        check_replacement(
            repo,
            OLD,
            {OLD: candidate(tmp_path, old_rules())},
            record(tmp_path, schema="other"),
        )
    with pytest.raises(ReplacementCheckError, match="ISO date"):
        check_replacement(
            repo,
            OLD,
            {OLD: candidate(tmp_path, old_rules())},
            record(tmp_path, [("limit_rate", "March 2026")]),
        )


def test_first_difference_after_effective_to():
    old = [("0001-01-01", None, "a")]
    new = [("0001-01-01", "2025-12-31", "a"), ("2026-01-01", None, "b")]
    assert first_difference(old, new) == "2026-01-01"
    ended = [("0001-01-01", "2025-12-31", "a")]
    assert first_difference(old, ended) == "2026-01-01"
    assert first_difference(old, old) is None


@pytest.mark.parametrize(
    "old_path",
    ["../outside.yaml", "/abs/x/y.yaml", "us-az/x.test.yaml", "us-az/x.yaml"],
)
def test_bad_old_paths(repo, tmp_path, old_path):
    with pytest.raises(ReplacementCheckError):
        check_replacement(repo, old_path, {old_path: candidate(tmp_path, old_rules())})


def test_cli_exit_codes(repo, tmp_path, capsys):
    same = candidate(tmp_path, old_rules())
    assert main(["--repo", str(repo), "--old", OLD, "--new", str(same)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    changed = candidate(tmp_path, [raised_rate(), *old_rules()[1:]], "c.yaml")
    assert main(["--repo", str(repo), "--old", OLD, "--new", f"{OLD}={changed}"]) == 1
    assert codes(json.loads(capsys.readouterr().out)) == ["undeclared_change"]
    assert (
        main(["--repo", str(repo), "--old", "missing/x/y.yaml", "--new", str(same)])
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().err)


def relation(arguments):
    return {
        "name": "member_of_unit",
        "kind": "data_relation",
        "data_relation": {"arity": len(arguments), "arguments": arguments},
    }


def test_relation_rule_interface_is_checked(repo, tmp_path):
    write(repo / OLD, module(*old_rules(), relation(["Person", "Household"])))
    write(
        repo / "us-az/policies/des/other/rel.yaml",
        module(
            rule("count", "len(member_of_unit)"),
            imports=["us-az:policies/des/ece/expanded#member_of_unit"],
        ),
    )
    new = candidate(tmp_path, [*old_rules(), relation(["Person", "TaxUnit", "X"])])
    report = check_replacement(repo, OLD, {OLD: new})
    assert "export_type_changed" in codes(report)
    assert report["exports"]["type_changed"][0]["fields"] == ["data_relation"]


def test_split_onto_an_existing_module_is_refused(repo, tmp_path):
    victim = "us-az/policies/des/ece/member.yaml"
    new = {
        victim: candidate(tmp_path, old_rules(), "v.yaml"),
        "us-az/manual/des/new/block-1.yaml": candidate(tmp_path, [], "n.yaml"),
    }
    assert "new_path_exists" in codes(check_replacement(repo, OLD, new))


@pytest.mark.parametrize(
    "before, after",
    [('"single"', "single"), ('status == "a  b"', 'status == "a b"')],
)
def test_string_literals_are_compared_exactly(repo, tmp_path, before, after):
    write(repo / OLD, module(*old_rules(), rule("status_rule", before)))
    new = candidate(tmp_path, [*old_rules(), rule("status_rule", after)])
    report = check_replacement(repo, OLD, {OLD: new})
    assert codes(report) == ["undeclared_change"]


def test_from_and_to_version_keys(repo, tmp_path):
    versions = [{"from": "2020-01-01", "to": "2025-12-31", "formula": "1"}]
    write(repo / OLD, module(*old_rules(), rule("amount", None, versions=versions)))
    same = candidate(tmp_path, [*old_rules(), rule("amount", None, versions=versions)])
    assert check_replacement(repo, OLD, {OLD: same})["passed"] is True
    longer = [{"from": "2020-01-01", "formula": "1"}]
    report = check_replacement(
        repo,
        OLD,
        {
            OLD: candidate(
                tmp_path,
                [*old_rules(), rule("amount", None, versions=longer)],
                "l.yaml",
            )
        },
    )
    assert report["behavior_changes"][0]["first_difference"] == "2026-01-01"


def test_two_versions_with_one_start_date_are_refused(repo, tmp_path):
    versions = [
        {"effective_from": "0001-01-01", "formula": "1.85"},
        {"effective_from": "0001-01-01", "formula": "2.0"},
    ]
    rules = [
        rule("limit_rate", None, kind="parameter", versions=versions),
        *old_rules()[1:],
    ]
    with pytest.raises(ReplacementCheckError, match="one start date"):
        check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})


def test_declared_change_of_a_removed_rule(repo, tmp_path):
    rules = [item for item in old_rules() if item["name"] != "unread_flag"]
    path = record(tmp_path, [("unread_flag", "2026-03-01")], removed=["unread_flag"])
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)}, path)
    assert codes(report) == ["declared_change_removed"]


# Round 2: each test below is a false pass or a crash that a review reproduced.


@pytest.mark.parametrize(
    "spelling",
    [
        "./" + OLD,
        OLD.replace("us-az/", "us-az//", 1),
        OLD.replace("us-az/", "us-az/./", 1),
        OLD.replace("us-az/", "US-AZ/", 1),
        OLD.replace("expanded", "Expanded"),
        OLD + " ",
    ],
)
def test_non_canonical_old_path_is_refused(repo, tmp_path, spelling, capsys):
    """A spelling that finds the file but misses the scope compare is refused."""

    rules = [item for item in old_rules() if item["name"] != "ece"]
    new = candidate(tmp_path, rules)
    with pytest.raises(ReplacementCheckError, match="canonical|exact path"):
        check_replacement(repo, spelling, {spelling: new})
    assert main(["--repo", str(repo), "--old", spelling, "--new", str(new)]) == 2
    assert "error" in json.loads(capsys.readouterr().err)


def test_symlinked_old_path_is_refused(repo, tmp_path):
    link = repo / "us-az/policies/des/ece/link.yaml"
    link.symlink_to(repo / OLD)
    path = "us-az/policies/des/ece/link.yaml"
    with pytest.raises(ReplacementCheckError, match="exact path"):
        check_replacement(repo, path, {path: candidate(tmp_path, old_rules())})


def module_change_record(tmp_path, *fields, **entries):
    """Declare ``fields``; ``entries`` gives the exact entries of one field."""

    changes = [
        {"field": field, "reason": "source moved", **entries} for field in fields
    ]
    return record(tmp_path, module_changes=changes)


def test_module_imports_are_compared(repo, tmp_path):
    write(repo / OLD, module(*old_rules(), imports=["us:statutes/a#x"]))
    same = candidate(tmp_path, old_rules(), imports=["us:statutes/a#x"])
    assert check_replacement(repo, OLD, {OLD: same})["passed"] is True
    moved = candidate(tmp_path, old_rules(), "m.yaml", imports=["us:statutes/b#x"])
    report = check_replacement(repo, OLD, {OLD: moved})
    assert codes(report) == ["undeclared_module_change"]
    assert "us:statutes/a#x" in report["issues"][0]["message"]
    assert "us:statutes/b#x" in report["issues"][0]["message"]
    assert [item["field"] for item in report["module_changes"]] == ["imports"]
    declared = module_change_record(
        tmp_path, "imports", added=["us:statutes/b#x"], removed=["us:statutes/a#x"]
    )
    assert check_replacement(repo, OLD, {OLD: moved}, declared)["passed"] is True
    absent = check_replacement(repo, OLD, {OLD: same}, declared)
    assert codes(absent) == ["declared_module_change_absent"]


def test_dropped_module_imports_are_a_change(repo, tmp_path):
    write(repo / OLD, module(*old_rules(), imports=["us:statutes/a#x"]))
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, old_rules())})
    assert codes(report) == ["undeclared_module_change"]


@pytest.mark.parametrize(
    "field, value",
    [("entity", "Person"), ("dtype", "Count"), ("period", "Year"), ("unit", "USD")],
)
def test_module_inputs_are_compared(repo, tmp_path, field, value):
    declared_input = {
        "name": "gross_income",
        "entity": "Household",
        "dtype": "Money",
        "period": "Month",
        "description": "supplied",
    }
    write(repo / OLD, module(*old_rules(), inputs=[declared_input]))
    reworded = dict(declared_input, description="supplied by the caller")
    same = candidate(tmp_path, old_rules(), inputs=[reworded])
    assert check_replacement(repo, OLD, {OLD: same})["passed"] is True
    changed = candidate(
        tmp_path, old_rules(), "i.yaml", inputs=[dict(declared_input, **{field: value})]
    )
    report = check_replacement(repo, OLD, {OLD: changed})
    assert codes(report) == ["undeclared_module_change"]
    assert "gross_income" in report["issues"][0]["message"]
    declared = module_change_record(tmp_path, "inputs", changed=["gross_income"])
    assert check_replacement(repo, OLD, {OLD: changed}, declared)["passed"] is True
    dropped = candidate(tmp_path, old_rules(), "d.yaml")
    assert codes(check_replacement(repo, OLD, {OLD: dropped})) == [
        "undeclared_module_change"
    ]


def test_other_module_fields_are_compared(repo, tmp_path):
    new = candidate(tmp_path, old_rules(), status="deferred")
    report = check_replacement(repo, OLD, {OLD: new})
    assert codes(report) == ["undeclared_module_change"]
    assert report["module_changes"][0]["field"] == "module.status"
    # Provenance metadata is not a module change.
    metadata = candidate(tmp_path, old_rules(), "p.yaml", proof_validation={"a": 1})
    assert check_replacement(repo, OLD, {OLD: metadata})["passed"] is True


@pytest.mark.parametrize(
    "entry",
    [
        "policies/des/ece/expanded.yaml",
        "./policies/des/ece/expanded",
        "policies/des/ece//expanded",
        "policies/des/ece/Expanded",
        "policies/des/ece/no-such-module",
        {"path": "policies/des/ece/expanded"},
    ],
)
def test_unresolvable_scope_entry_is_refused(repo, tmp_path, entry):
    """A scope entry the check cannot resolve may scope the old module."""

    spec = repo / "programs/us-az/snap/fy-2026.yaml"
    program = yaml.safe_load(spec.read_text())
    program["scope"]["state"] = [entry, "policies/des/ece/member"]
    write(spec, program)
    rules = [item for item in old_rules() if item["name"] != "ece"]
    with pytest.raises(ReplacementCheckError, match="scope"):
        check_replacement(
            repo,
            OLD,
            {OLD: candidate(tmp_path, rules)},
            record(tmp_path, removed=["ece"]),
        )


@pytest.mark.parametrize(
    "name, definer",
    [
        ("gross_income", "programs/us-az/snap/fy-2026.yaml"),
        ("snap_eligible", "programs/us-az/snap/fy-2026.yaml"),
        ("member_rule", "us-az/policies/des/ece/member.yaml"),
        ("federal_rule", "us/regulations/7-cfr/273/9.yaml"),
    ],
)
def test_added_rule_name_must_not_collide(repo, tmp_path, name, definer):
    """The composer binds by name; a new rule must not take a bound name."""

    new = candidate(tmp_path, [*old_rules(), rule(name, "within_limit")])
    report = check_replacement(repo, OLD, {OLD: new})
    assert codes(report) == ["added_name_collides"]
    assert definer in report["issues"][0]["message"]
    fresh = candidate(tmp_path, [*old_rules(), rule("fresh_name", "within_limit")])
    assert check_replacement(repo, OLD, {OLD: fresh})["passed"] is True


def test_duplicate_yaml_keys_are_refused(repo, tmp_path):
    text = yaml.safe_dump(module(*old_rules()), sort_keys=False)
    # safe_load keeps the last value: the rule list below would win silently.
    shadow = yaml.safe_dump({"rules": old_rules()[:1]}, sort_keys=False)
    doubled = tmp_path / "doubled.yaml"
    doubled.write_text(text + shadow)
    assert len(yaml.safe_load(doubled.read_text())["rules"]) == 1
    with pytest.raises(ReplacementCheckError, match="repeats a YAML key"):
        check_replacement(repo, OLD, {OLD: doubled})
    nested = tmp_path / "nested.yaml"
    nested.write_text(
        text.replace("  kind: parameter\n", "  kind: x\n  kind: parameter\n")
    )
    with pytest.raises(ReplacementCheckError, match="repeats a YAML key"):
        check_replacement(repo, OLD, {OLD: nested})
    bad_record = record(tmp_path)
    bad_record.write_text(bad_record.read_text() + "replaced: other\n")
    with pytest.raises(ReplacementCheckError, match="repeats a YAML key"):
        check_replacement(
            repo, OLD, {OLD: candidate(tmp_path, old_rules())}, bad_record
        )
    (repo / OLD).write_text(text + shadow)
    with pytest.raises(ReplacementCheckError, match="repeats a YAML key"):
        check_replacement(repo, OLD, {OLD: candidate(tmp_path, old_rules())})


def test_declared_effective_from_must_equal_the_first_difference(repo, tmp_path):
    rules = [raised_rate(), *old_rules()[1:]]
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, [("limit_rate", "2026-01-01")]),
    )
    assert codes(report) == ["declared_effective_from_too_early"]
    assert "2026-03-01" in report["issues"][0]["message"]


def test_non_string_rule_key_is_a_structured_refusal(repo, tmp_path, capsys):
    rules = [{**old_rules()[0], 7: "x"}, *old_rules()[1:]]
    new = candidate(tmp_path, rules)
    with pytest.raises(ReplacementCheckError, match="not a string"):
        check_replacement(repo, OLD, {OLD: new})
    assert main(["--repo", str(repo), "--old", OLD, "--new", str(new)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not a string" in json.loads(captured.err)["error"]


def test_unexpected_failure_is_a_structured_refusal(
    repo, tmp_path, capsys, monkeypatch
):
    def broken(*args, **kwargs):
        raise TypeError("boom")

    monkeypatch.setattr(replacement_check, "check_replacement", broken)
    new = candidate(tmp_path, old_rules())
    assert main(["--repo", str(repo), "--old", OLD, "--new", str(new)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "TypeError" in json.loads(captured.err)["error"]


@pytest.mark.parametrize(
    "imports",
    [
        [{"module": "policies/des/ece/expanded"}],
        [7],
        "policies/des/ece/expanded",
        ["us-az:policies/des/ece/./expanded"],
        ["us-az:policies/des/ece/expanded.yaml"],
        ["policies/des/ece/expanded#"],
    ],
)
def test_unreadable_import_entry_is_refused(repo, tmp_path, imports):
    """An import entry that is skipped could hide a reader of the old module."""

    payload = module(rule("uses_ece", "ece and something_else"))
    payload["imports"] = imports
    write(repo / "us-az/policies/des/other/odd.yaml", payload)
    rules = [item for item in old_rules() if item["name"] != "ece"]
    with pytest.raises(ReplacementCheckError, match="import"):
        check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})


def test_importer_is_found_without_the_module_path_in_its_text(repo, tmp_path):
    importer = repo / "us-az/policies/des/other/escaped.yaml"
    write(importer, module(rule("uses_unread", "unread_flag")))
    importer.write_text(
        importer.read_text()
        + 'imports: ["us-az:policies/des/ece/\\x65xpanded#unread_flag"]\n'
    )
    assert "ece/expanded" not in importer.read_text()
    rules = [item for item in old_rules() if item["name"] != "unread_flag"]
    report = check_replacement(
        repo,
        OLD,
        {OLD: candidate(tmp_path, rules)},
        record(tmp_path, removed=["unread_flag"]),
    )
    assert "export_removed" in codes(report)


def test_not_checked_names_other_jurisdiction_repositories(repo, tmp_path):
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, old_rules())})
    assert report["not_checked"] == list(NOT_CHECKED)
    assert any(
        "rulespec-us-tn" in item and "rulespec-us-nc" in item for item in NOT_CHECKED
    )


def tree_state(root):
    return {
        path.relative_to(root).as_posix(): (
            "dir"
            if path.is_dir()
            else (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
        )
        for path in sorted(root.rglob("*"))
    }


def test_repo_tree_is_byte_unchanged_after_a_run(repo, tmp_path, capsys):
    """The check is read-only: pass, issue, split and refusal leave no trace."""

    same = candidate(tmp_path, old_rules())
    changed = candidate(tmp_path, [raised_rate(), *old_rules()[1:]], "c.yaml")
    block = "us-az/manual/des/faa5/ece/block-4.yaml"
    declared = record(tmp_path, [("limit_rate", "2026-03-01")])
    before = tree_state(repo)
    base = ["--repo", str(repo), "--old", OLD]
    assert main([*base, "--new", str(same)]) == 0
    assert main([*base, "--new", str(changed)]) == 1
    assert main([*base, "--new", str(changed), "--record", str(declared)]) == 0
    assert main([*base, "--new", f"{block}={same}"]) == 1
    assert main([*base, "--new", str(tmp_path / "missing.yaml")]) == 2
    assert main(["--repo", str(repo), "--old", "./" + OLD, "--new", str(same)]) == 2
    capsys.readouterr()
    assert tree_state(repo) == before


# Review round 3.


@pytest.mark.parametrize("key", ["auto_gate_outputs", "acknowledged_incomplete"])
def test_rule_named_only_in_a_program_name_list_has_a_reader(repo, tmp_path, key):
    """A program spec reads every rule it names, not only its outputs."""

    spec = repo / "programs/us-az/snap/fy-2026.yaml"
    program = yaml.safe_load(spec.read_text())
    program[key] = ["unread_flag"]
    write(spec, program)
    removed = candidate(
        tmp_path, [item for item in old_rules() if item["name"] != "unread_flag"]
    )
    declared = record(tmp_path, removed=["unread_flag"])
    report = check_replacement(repo, OLD, {OLD: removed}, declared)
    assert "export_removed" in codes(report)
    assert report["readers"]["unread_flag"] == ["programs/us-az/snap/fy-2026.yaml"]
    retyped = candidate(
        tmp_path,
        [
            rule("unread_flag", "within_limit", entity="Person")
            if item["name"] == "unread_flag"
            else item
            for item in old_rules()
        ],
        "retyped.yaml",
    )
    declared = record(
        tmp_path, interface_changes=[{"name": "unread_flag", "reason": "x"}]
    )
    report = check_replacement(repo, OLD, {OLD: retyped}, declared)
    assert "export_type_changed" in codes(report)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"gated_outputs": ["unread_flag"]}, "program keys"),
        ({"outputs": [{"name": "snap_eligible"}]}, "outputs"),
        ({"auto_gate_outputs": "snap_eligible"}, "auto_gate_outputs"),
        (
            {"acknowledged_incomplete": {"snap_eligible": "x"}},
            "acknowledged_incomplete",
        ),
        ({"transformations": {"name": "x"}}, "transformations"),
        (
            {"transformations": [{"name": "x", "pattern": "all_of", "inputs": ["y"]}]},
            "transformation x has keys",
        ),
        ({"scope": {"local": ["policies/des/ece/expanded"]}}, "scope key"),
    ],
)
def test_unknown_program_shape_is_refused(repo, tmp_path, change, match):
    # An unrelated spec: the refusal does not depend on the scope.
    spec = repo / "programs/us-az/tanf/fy-2026.yaml"
    write(spec, {**yaml.safe_load(spec.read_text()), **change})
    with pytest.raises(ReplacementCheckError, match=match):
        check_replacement(repo, OLD, {OLD: candidate(tmp_path, old_rules())})


def with_metadata(metadata):
    return [
        dict(item, metadata=metadata) if item["name"] == "unread_flag" else item
        for item in old_rules()
    ]


def test_rule_metadata_private_is_an_interface_change(repo, tmp_path):
    """``metadata.private`` changes visibility; the proof block is provenance."""

    proof = {"atoms": [{"path": "versions[0].formula", "kind": "source"}]}
    write(repo / OLD, module(*with_metadata({"proof": proof})))
    reproved = candidate(tmp_path, with_metadata({"proof": {"atoms": []}}))
    assert check_replacement(repo, OLD, {OLD: reproved})["passed"] is True
    no_proof = candidate(tmp_path, old_rules(), "n.yaml")
    assert check_replacement(repo, OLD, {OLD: no_proof})["passed"] is True
    private = candidate(
        tmp_path, with_metadata({"proof": proof, "private": True}), "p.yaml"
    )
    report = check_replacement(repo, OLD, {OLD: private})
    assert codes(report) == ["undeclared_interface_change"]
    assert report["exports"]["type_changed"][0]["fields"] == ["metadata"]
    noted = candidate(
        tmp_path, with_metadata({"proof": proof, "caller_obligations": "x"}), "c.yaml"
    )
    assert codes(check_replacement(repo, OLD, {OLD: noted})) == [
        "undeclared_interface_change"
    ]


def test_rule_metadata_private_on_a_read_rule_is_a_type_change(repo, tmp_path):
    rules = [
        dict(item, metadata={"private": True}) if item["name"] == "ece" else item
        for item in old_rules()
    ]
    report = check_replacement(repo, OLD, {OLD: candidate(tmp_path, rules)})
    assert codes(report) == ["export_type_changed"]


def test_import_declaration_must_list_the_exact_entries(repo, tmp_path):
    write(repo / OLD, module(*old_rules(), imports=["us:statutes/a#x"]))
    moved = candidate(
        tmp_path, old_rules(), imports=["us:statutes/b#x", "us:statutes/c#y"]
    )
    bare = module_change_record(tmp_path, "imports")
    assert codes(check_replacement(repo, OLD, {OLD: moved}, bare)) == [
        "module_change_declaration_inexact"
    ]
    partial = module_change_record(
        tmp_path, "imports", added=["us:statutes/b#x"], removed=["us:statutes/a#x"]
    )
    report = check_replacement(repo, OLD, {OLD: moved}, partial)
    assert codes(report) == ["module_change_declaration_inexact"]
    assert "us:statutes/c#y" in report["issues"][0]["message"]
    exact = module_change_record(
        tmp_path,
        "imports",
        added=["us:statutes/c#y", "us:statutes/b#x"],
        removed=["us:statutes/a#x"],
    )
    assert check_replacement(repo, OLD, {OLD: moved}, exact)["passed"] is True


def test_input_declaration_must_list_the_exact_entries(repo, tmp_path):
    def declared_input(name, **extra):
        return {
            "name": name,
            "entity": "Household",
            "dtype": "Money",
            "period": "Month",
            **extra,
        }

    write(
        repo / OLD,
        module(*old_rules(), inputs=[declared_input("a"), declared_input("b")]),
    )
    new = candidate(
        tmp_path,
        old_rules(),
        inputs=[declared_input("a", dtype="Count"), declared_input("c")],
    )
    for entries in (
        {},
        {"changed": ["a"]},
        {"changed": ["a"], "added": ["c"]},
        {"changed": ["a", "b"], "added": ["c"]},
        {"changed": ["a"], "added": ["c"], "removed": ["b", "z"]},
    ):
        inexact = module_change_record(tmp_path, "inputs", **entries)
        assert codes(check_replacement(repo, OLD, {OLD: new}, inexact)) == [
            "module_change_declaration_inexact"
        ]
    exact = module_change_record(
        tmp_path, "inputs", changed=["a"], added=["c"], removed=["b"]
    )
    assert check_replacement(repo, OLD, {OLD: new}, exact)["passed"] is True


@pytest.mark.parametrize(
    "item",
    [
        {"field": "imports", "reason": "x", "changed": ["us:statutes/a#x"]},
        {"field": "imports", "reason": "x", "added": "us:statutes/a#x"},
        {"field": "inputs", "reason": "x", "added": ["a", "a"]},
        {"field": "format", "reason": "x", "added": ["a"]},
    ],
)
def test_module_change_declaration_it_cannot_read_is_refused(repo, tmp_path, item):
    declared = record(tmp_path, module_changes=[item])
    with pytest.raises(ReplacementCheckError, match="module change"):
        check_replacement(repo, OLD, {OLD: candidate(tmp_path, old_rules())}, declared)


@pytest.mark.parametrize(
    "old, new",
    [
        ("1.85", "1.8_5"),
        ("3", "03"),
        ("1000", "1_000"),
        ("1000", "1e3"),
        ("1", "+1"),
        ("0.5", ".5"),
        ("8", "010"),
    ],
)
@pytest.mark.parametrize("quoted", [True, False])
def test_number_outside_the_strict_grammar_is_compared_as_text(
    repo, tmp_path, old, new, quoted
):
    """``Decimal()`` and YAML 1.1 accept spellings another reader can read otherwise."""

    def text(value):
        value = f"'{value}'" if quoted else value
        return (
            "format: rulespec/v1\nmodule:\n  summary: x\nrules:\n"
            "- name: limit_rate\n  kind: parameter\n  dtype: Rate\n"
            f"  versions:\n  - effective_from: 0001-01-01\n    formula: {value}\n"
        )

    (repo / OLD).write_text(text(old))
    same = tmp_path / "same.yaml"
    same.write_text(text(old))
    assert check_replacement(repo, OLD, {OLD: same})["passed"] is True
    other = tmp_path / "other.yaml"
    other.write_text(text(new))
    assert codes(check_replacement(repo, OLD, {OLD: other})) == ["undeclared_change"]


@pytest.mark.parametrize("old, new", [("1.85", "1.850"), ("3", "3.0"), ("-2", "-2.00")])
def test_plain_decimals_with_one_value_are_equal(repo, tmp_path, old, new):
    for index, (a, b) in enumerate([(old, new), (f"'{old}'", new), (old, f"'{new}'")]):
        write(repo / OLD, module(rule("limit_rate", "0", kind="parameter")))
        text = (repo / OLD).read_text()
        (repo / OLD).write_text(text.replace("formula: '0'", f"formula: {a}"))
        same = tmp_path / f"same-{index}.yaml"
        same.write_text(text.replace("formula: '0'", f"formula: {b}"))
        assert f"formula: {b}" in same.read_text()
        assert check_replacement(repo, OLD, {OLD: same})["passed"] is True


def test_unknown_input_key_is_compared(repo, tmp_path):
    """Only ``description`` is prose; ``source``, ``metadata``, ``summary`` are not."""

    declared_input = {"name": "gross_income", "entity": "Household", "dtype": "Money"}
    for key in ("source", "metadata", "summary", "default"):
        write(
            repo / OLD,
            module(*old_rules(), inputs=[dict(declared_input, **{key: "a"})]),
        )
        changed = candidate(
            tmp_path, old_rules(), inputs=[dict(declared_input, **{key: "b"})]
        )
        report = check_replacement(repo, OLD, {OLD: changed})
        assert codes(report) == ["undeclared_module_change"], key


def split_files(tmp_path):
    return {
        "us-az/manual/des/faa5/ece/block-4.yaml": candidate(
            tmp_path, old_rules(), "block-4.yaml"
        )
    }


@pytest.fixture
def unread_repo(repo):
    """The old module with no importer and no scoping program spec."""

    (repo / "us-az/policies/des/ece/benefit.yaml").unlink()
    (repo / "us-az/policies/des/other/user.yaml").unlink()
    (repo / "programs/us-az/snap/fy-2026.yaml").unlink()
    return repo


def test_split_with_no_importers_and_no_registry_entry_passes(unread_repo, tmp_path):
    write(
        unread_repo / "known-validation-gaps.yaml",
        {"validate_failures": {"us-az/policies/des/ece/member.yaml": {"active": {}}}},
    )
    report = check_replacement(unread_repo, OLD, split_files(tmp_path))
    assert report["passed"] is True
    assert report["mechanical_updates"] == []


def test_split_reports_registries_and_companion_test(unread_repo, tmp_path):
    """Files that still name the old path are issues, not only ``not_checked``."""

    write(
        unread_repo / "known-validation-gaps.yaml",
        {"validate_failures": {OLD: {"active": {"fingerprint": "sha256:0"}}}},
    )
    write(
        unread_repo / "oracle-coverage-pending.yaml",
        {
            "version": 1,
            "entries": [
                {"legal_id": "us-az:policies/des/ece/expanded#ece"},
                {"legal_id": "us-az:policies/des/ece/expanded#within_limit"},
                {"legal_id": "us-az:policies/des/ece/expanded-other#ece"},
            ],
        },
    )
    companion = OLD.removesuffix(".yaml") + ".test.yaml"
    write(unread_repo / companion, [{"name": "case", "output": {}}])
    report = check_replacement(unread_repo, OLD, split_files(tmp_path))
    assert codes(report) == [
        "companion_test_not_moved",
        "registry_not_updated",
        "registry_not_updated",
    ]
    assert report["mechanical_updates"] == [
        {"kind": "companion_test", "path": companion},
        {"kind": "registry_entry", "path": "known-validation-gaps.yaml", "count": 1},
        {"kind": "registry_entry", "path": "oracle-coverage-pending.yaml", "count": 2},
    ]
    # A same-path replacement keeps the path, so nothing has to move.
    same = check_replacement(
        unread_repo, OLD, {OLD: candidate(tmp_path, old_rules(), "same.yaml")}
    )
    assert same["passed"] is True


def test_registry_the_check_cannot_parse_is_refused(unread_repo, tmp_path):
    (unread_repo / "known-validation-gaps.yaml").write_text("a: 1\na: 2\n")
    with pytest.raises(ReplacementCheckError, match="registry file"):
        check_replacement(unread_repo, OLD, split_files(tmp_path))
