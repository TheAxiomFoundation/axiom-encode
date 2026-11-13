## Summary

<!-- What changes and why. Link the issue. -->

Validation effect: <none|loosen|tighten>
<!--
Required (#2094). One of:
- none: RuleSpec validation results cannot change.
- loosen: some RuleSpec file can now pass that failed before.
- tighten: some RuleSpec file can now fail that passed before.
For loosen or tighten, add one line with the changed check ids, or unknown:
Changed checks: <check-id>, <check-id>
-->

## Encoding linkage

<!--
Both lines are required; CI (PR linkage) checks them. Write `none` when the
change fixes no particular encoding or run (generic work), so counts stay exact.
The placeholders below fail the check until you replace them.

Encoding-Target: the corpus citation path(s) of the encoding(s) this PR fixes,
comma-separated, for example us/regulation/7/273/7. A citation, not a .yaml path.

Fixes-Run: the run(s) that showed the defect, comma-separated:
  - a targeted-signed-reencode run: its id (42409703021) or its actions/runs URL;
    CI checks that it is a workflow_dispatch run on main and that
    Encoding-Target lists its citation (the last word of the run name, even
    when the fix is for a dependent);
  - a local encode: local:<encoding_runs id>, for example local:1a2b3c4d.
-->

Encoding-Target: REPLACE-with-citation-paths-or-none
Fixes-Run: REPLACE-with-run-ids-or-none

## Checks

<!-- Commands run and the review-fix cycle outcome. -->
