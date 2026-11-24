"""The replacement dry-run check: interface, program bindings, declared change."""

import json

import pytest
import yaml

from axiom_encode.replacement_check import (
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


def module(*rules, imports=None):
    payload = {"format": "rulespec/v1", "module": {"summary": "x"}}
    if imports:
        payload["imports"] = imports
    payload["rules"] = list(rules)
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


def candidate(tmp_path, rules, name="new.yaml", imports=None):
    return write(tmp_path / "candidates" / name, module(*rules, imports=imports))


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
    "name, field, value, refused",
    [
        ("limit_rate", "dtype", "Money", True),
        ("ece", "entity", "Person", True),
        ("within_limit", "period", "Year", True),
        ("ece", "unit", "USD", True),
        ("ece", "visibility", "private", True),
        ("unread_flag", "entity", "Person", True),
    ],
)
def test_interface_change(repo, tmp_path, name, field, value, refused):
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
    assert report["passed"] is (not refused)


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
        block5: candidate(tmp_path, old_rules()[2:], "b5.yaml"),
    }
    report = check_replacement(repo, OLD, new)
    assert report["form"] == "split_or_move"
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
