# Structural reading of lists in the gates-collapsed check

Status: implemented behind a switch that is **off**. With the switch off the
validator is the flat reader, unchanged. This page describes what the switch
turns on, what it guarantees, and what it does not cover. It is Option 2 of
issue #2300, delivered for lists; two source forms are left as follow-ups.

## The problem

`complete-source-unit:source-explicit-conditions/gates-collapsed` counts the
simultaneous facts of a source condition and wants a distinct input for each
fact on every formula path. It reads flat text: a cue (`if`, `when`,
`provided that`, `unless`) opens a clause, and every bullet after it is one more
simultaneous fact. The clause is found through a formula proof excerpt.

That gives false refusals:

- A list of alternatives ("when any of the following occur:") is read as a
  conjunction. A faithful `a or b or c` is refused.
- A `NOTE`, or a `when` sentence in one item, runs on into the items after it.
- A clause runs past the list into the next paragraph when the last bullet has
  no full stop.

It also gives false passes:

- A list is seen only when a proof excerpt quotes its chapeau. With item-only
  excerpts a candidate that drops a conjunct has no finding.
- For a list of alternatives without verb phrases, `a and b and c`, a negated
  rule and a constant rule pass, because the check counts inputs on a path and
  does not read the `or`.

## The switch

`axiom_encode.harness.source_list_structure.STRUCTURAL_GATE_READING_ENABLED`,
a module-level constant, default `False`. While it is `False` no function of
the structural reading is called
(`test_the_structural_code_is_not_reached_while_the_switch_is_off`), and the
replay snapshot is unchanged.

Turning it on changes validator decisions in both directions. That change
needs its own pull request with `Validation effect:` declared for it, a
regenerated replay snapshot, and an owner decision.

## The tree

`parse_source_lists(text)` reads the scanned unit text into blocks and lists.

- A **block** is one physical line; a line that directly follows another line
  (no blank line) continues that block. A block is an `item` (it starts with
  `●` or `•` and one space), a `note` (`NOTE` and a capital letter), an
  `example`, or a `paragraph`.
- A **list** is a paragraph that ends with a colon (the chapeau), directly
  followed by marked items with one marker. Between items it holds `NOTE` and
  `Example` blocks, and, after an item that ends with a colon, unmarked
  paragraphs (the children of that item; the corpus prints nested items without
  a marker). The list ends at the first unmarked paragraph after an item that
  does not end with a colon.
- The reader **declines** a layout that it cannot state: indented lines, tabs,
  carriage returns, a marker without its space (no list at all); a marked item
  that ends with a colon and has no unmarked child, two markers in one list,
  more than 12 items (the list is `unrecognised`). A marked item is never a
  list head.

## Closed evidence in the chapeau

A list is a list of **alternatives** only when its chapeau has one of two
forms and holds the word "following" exactly once:

| Form | Pattern | Example (constructed) |
|---|---|---|
| cue form | `when` or `if`, directly followed by `any`, `one or more`, `either`, `any one`, `at least one` or `one`, then `of the following`, an optional noun (`conditions`, `situations`, `circumstances`, `criteria`, `events`, `reasons`), an optional affirmative verb (`occur(s)`, `apply/applies`, `exist(s)`, `is/are met`, `is/are true`), and the colon at the end | "A permit must be renewed when any of the following occur:" |
| subject form | the whole chapeau is `Any of the following <one clause>:` with a capital `A` and no inner sentence punctuation | "Any of the following fees are not refundable:" |

Every other chapeau is `conjunctive` ("all|each|both of the following") or
`unrecognised`. Both are read flat: every bullet is a gate. There is no list of
words that turn a reading off; a wording outside the two forms simply is not an
alternatives list ("when each one of", "when not any of", "do not apply",
"unless any of", "when the holder has any of").

## Clause bounds on the tree

Two rules. Each applies only to an alternatives list.

1. **A clause inside the list stays in its own block.** When the cue stands in
   an item, a note or an unmarked child of the list, and the text up to the
   next marked item ends with a full stop, the clause ends where that item
   starts. The items after it are alternatives of the chapeau; they cannot be
   conjuncts of a sentence inside another member. A cue that stands before the
   list's chapeau is not cut: the flat reading stays.
2. **A clause of the chapeau ends with the list.** The end of the last item is
   treated as the end of a sentence under the flat reader's own rule (a capital
   letter, `(` or a numbered label follows), and only when the items end alike
   (all with a full stop, or all with a letter, a digit or `)`) and the last
   item has no unmarked children. A rule that quotes the paragraph after the
   list is then read on that paragraph alone.

An excerpt that crosses one of these bounds keeps the flat reading.

## What is verified for a list of alternatives

For a clause whose cue is in a cue-form chapeau:

- The facts of the chapeau before "when|if any of the following" are read by
  the flat reader and stay simultaneous facts. Every true path of the formula
  must give each its own input.
- Each marked item is one alternative. Its facts are read by the flat reader
  (a conjunction inside an item gives several facts); an item without verb
  wording gives one nominal fact.
- The formula is **evaluated**. `signed_granting_paths` (the closed Boolean
  reader that the physical-allowance check already uses) gives the true paths
  over the terminal inputs, through derived Boolean rules. With the chapeau
  inputs true, and for every value of every input that matches no source fact:
  - the formula is false while no input of any item is true;
  - each item has distinct inputs, one for each of its facts, that make the
    formula true while every other item input is false, and it is false again
    when one of them is false;
  - no item's inputs are contained in another item's.

This is why naming an input is not enough. The earlier registry design with a
group gate (#2216, review round 4) was satisfied by `... or L or (A and False)`,
because it looked at the names that a formula reaches. Here `A` in a dead arm
changes no value of the formula, so the item has no input that makes the rule
true alone.

Refused with the switch on (tests in
`tests/test_source_gates_structural_reading.py`): a dropped alternative, `and`
for `or`, a negated alternative, a negated rule, a constant rule, an
alternative in a dead arm or next to a constant, an alternative that works only
together with another one, one input for several alternatives, an invented
condition on the rule or on one alternative, an invented extra alternative, a
dropped fact of the chapeau, a dropped conjunct inside an item.

When the formula is outside the closed reader (a rule that is not Boolean, a
comparison, a function call, more than 16 inputs, versions that change inside
the period), the clause keeps the flat reading.

## Binding by structure

The flat reader sees a list only through an excerpt of its chapeau. With the
switch on, a rule whose formula proof stands inside a marked item is bound to
the list when the chapeau puts the items under its own cue by a closed form:
the cue form of alternatives, `when|if all|each of the following ...:`, or
`when|if <1-4 words> meets|satisfies all|each|at least N|N or more of the
following ...:`. When no rule quotes the chapeau, one bound rule, or a rule
that uses a bound rule, must pass the reading of the chapeau clause. Else the
list has a finding.

"When X, all of the following apply:" does not bind: its items are
consequences of X. A rule for each item, combined by a rule that quotes nothing
in the list, passes through the combining rule.

## Invariants

| | Invariant | Tests |
|---|---|---|
| (a) | Fail closed: without a closed chapeau form and a readable layout, every clause is read flat. Every relaxation is a positive form. | `test_a_text_without_a_closed_chapeau_form_is_read_exactly_as_flat` (property), `test_a_chapeau_outside_the_closed_forms_gives_the_flat_result`, `test_any_other_chapeau_is_not_read_as_alternatives`, the layout tests |
| (b) | No new escape: a list of alternatives is verified on the value of the formula. | `test_a_wrong_encoding_of_the_alternatives_is_refused`, `test_a_wrong_dependency_does_not_stand_for_an_alternative`, `test_a_conjunction_inside_one_alternative_is_demanded`, `test_a_fact_of_the_chapeau_stays_a_simultaneous_fact` |
| (c) | A rule that quotes only items is read against the list (closed chapeau forms). | `test_a_dropped_conjunct_is_refused_when_only_items_are_quoted`, `test_item_only_proof_of_a_list_of_alternatives_is_verified`, `test_one_rule_for_each_item_*` |
| (d) | A conjunctive or unrecognised list is refused at least as often as on the flat reading. | `test_a_conjunctive_list_is_refused_at_least_as_often_as_on_the_flat_reading` |

## Reviewed alternative lists

The registry (`docs/reviewed-alternative-lists.md`) is unchanged and stays the
stronger exemption. A unit whose exact text has a registry entry is not read
structurally at all: every clause of that unit is read as before, with the
entry. The same holds for a reviewed care source, a bound schedule and a
physical-allowance list, which have their own readers.

## Not covered

- A `NOTE` inside a list whose chapeau is not one of the two forms ("The
  following ... are allowable:"). The note still runs on.
- A fronted condition that is not ended before its main clause ("When X, and Y
  the worker does Z").
- A chapeau of alternatives in another wording ("may receive any of the
  following deductions"), and "either situation occurs".
- A conjunctive chapeau whose own words become a fact ("the payment meets all
  of the following"), and input names that do not match the source words.
- A list of alternatives under a negative output ("is not allowed when any of
  the following apply", encoded as `not (a or b)`): the reading wants a rule
  that is true when an alternative holds, so this encoding keeps a finding.
- An alternatives list without a cue in its chapeau is not verified. Rule 1
  removes the run-on from an inner `when`; the list itself is then unchecked,
  as it is on the flat reading when no item holds a cue.
- Lists with labels (`(a)`, `1.`) instead of markers, and nesting by indent.
- Residual assurance limits: an input is matched to a fact by its name, as
  everywhere in this check; a chapeau fact that the flat reader does not count
  is not demanded; binding is skipped above 8 lists or 6 bound rules.
