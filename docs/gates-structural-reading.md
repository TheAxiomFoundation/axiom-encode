# Structural reading of lists in the gates-collapsed check

Status: implemented behind a switch that is **off**. With the switch off the
validator is the flat reader, unchanged. This page describes what the switch
turns on, what it guarantees, and what it does not cover. It is Option 2 of
issue #2300, delivered for a narrow set of lists.

One rule governs every choice here: the structural reading relaxes the flat
reading only where it can positively verify the alternative reading **and**
the formula. In every other case it gives exactly the flat findings.

## The problem

`complete-source-unit:source-explicit-conditions/gates-collapsed` counts the
simultaneous facts of a source condition and wants a distinct input for each
fact on every formula path. It reads flat text: a cue (`if`, `when`,
`provided that`, `unless`) opens a clause, and every bullet after it is one more
simultaneous fact. The clause is found through a formula proof excerpt.

That gives false refusals (a list of alternatives is read as a conjunction; a
`NOTE` or a `when` sentence in one item runs on into the items after it) and
false passes (a list is seen only when a proof excerpt quotes its chapeau; for a
list of nominal alternatives `a and b and c`, a negated rule and a constant
rule pass, because the check counts inputs on a path and does not read the
`or`).

## The switch

`axiom_encode.harness.source_list_structure.STRUCTURAL_GATE_READING_ENABLED`,
a module-level constant, default `False`. While it is `False` no function of
the structural reading is called
(`test_the_structural_code_is_not_reached_while_the_switch_is_off`), and the
replay snapshot is unchanged. Turning it on changes validator decisions in both
directions. That needs its own pull request, a declared validation effect, a
regenerated replay snapshot, and an owner decision.

## The tree

`parse_source_lists(text)` reads the scanned unit text into blocks and lists.

- A **block** is one physical line; a line that directly follows another line
  continues that block. A block is an `item` (`●` or `•` and one space), a
  `note` (`NOTE` and a capital letter), an `example`, or a `paragraph`.
- A **list** is a paragraph that ends with a colon (the chapeau), directly
  followed by marked items with one marker. Between items it holds `NOTE` and
  `Example` blocks and, after an item that ends with a colon, unmarked
  paragraphs (the corpus prints nested items without a marker). The list ends
  at the first unmarked paragraph after an item that does not end with a colon.
- The reader **declines the whole unit** (no list at all) for indented lines,
  tabs, carriage returns, a marker without its space, and when any marked item
  belongs to no list. The last case is an interrupted list: a page line, a
  heading or any other paragraph between two items ends the list early, and
  the items after it would be lost.
- A list is `unrecognised` when a marked item ends with a colon and has no
  unmarked child, with two markers, or with more than 12 items. A marked item
  is never a list head.

## Closed evidence in the chapeau

A list is a list of **alternatives** only when its chapeau has one of two
forms and holds the word "following" exactly once:

| Form | Pattern |
|---|---|
| cue form | `when` or `if`, directly followed by `any`, `one or more`, `either`, `any one`, `at least one` or `one`, then `of the following`, an optional noun (`conditions`, `situations`, `circumstances`, `criteria`, `events`, `reasons`), an optional affirmative verb (`occur(s)`, `apply/applies`, `exist(s)`, `is/are met`, `is/are true`), and the colon |
| subject form | the whole chapeau is `Any of the following <one clause>:` with a capital `A` |

Every other chapeau is read flat.

## What an alternatives list changes

### 1. A sentence inside the list stays in its own block

When the cue and the proof excerpt both stand in an item, a note or an unmarked
child of the list, and the text up to the next marked item ends with a full
stop, the clause ends where that item starts. The later items are members of
the list; they are not facts of that sentence.

The sentence's own fact is still demanded. A cut clause that the flat reader
would skip (fewer than two facts) is read with one fact, and every true path of
the rule needs a **positive** input that matches it. A constant, a negated
input and an unrelated input are refused
(`test_the_one_fact_of_a_cut_note_is_still_demanded`).

A rule that quotes the chapeau is never cut. A cue before the list is never cut.

### 2. A verifiable list is verified as alternatives

A list is **verifiable** only when all of this holds; else the flat reading
stays for its clause:

- the chapeau has the cue form (or the subject form with a rule that quotes it
  while a cue stands in an item);
- the chapeau is a plain affirmative: before the cue it holds one of `must`,
  `may`, `is`, `are`, `can`, `shall`, `will`, no second cue or quantifier
  (`if`, `when`, `unless`, `even`, `each`, `every`, `both`, ...), and no word of
  the reversing list (`not`, `no`, `never`, `except`, `unless`, `without`,
  `but`, `cannot`, `n't`, `fail*`, `den*`, `refus*`, `invalid*`, `rarely`,
  `laps*`, `forfeit*`, ...);
- the list **ends the unit text**. A paragraph after the list can add to its
  condition ("Provided that ...", "The holder must also ...", "Residents alone
  qualify."), and no closed form tells that it does not;
- no item has unmarked children;
- every item is one plain alternative: it starts with a capital letter, holds
  words only, at most one `, and` joint and at most one `or`, a trailing
  `(g)` or `(See ...)` at most, no reversing word and no word that opens a
  qualifier (`if`, `when`, `provided`, `only`, `both`, `as`, `who`, `that`,
  `which`, `with`, `more`, `than`, ...);
- every part of a joint item is a fact that the flat reader counts.

The two word lists are a second layer. They are negative lists, and they cannot
be complete; the layer below them is positive (next list).

For a verifiable list the formula is **evaluated**. `signed_granting_paths`
(the closed Boolean reader of the physical-allowance check) gives the true
paths over the terminal inputs, through derived Boolean rules, with RuleSpec
`true` and `false` read as constants, at every date where a dependency version
changes. Then:

- **Every source word of an item is in the name of an input matched to it.**
  When each fact of the item has an input but a word is left over, the word may
  be a dropped qualifier: the flat reading stays.
- The formula is false while no input of any item is true, for every value of
  every unmatched input.
- Each item has distinct inputs, one for each of its facts, that make the
  formula true **whatever the other items' inputs are** (a plain `or`, not
  "exactly one" and not "unless another holds"), and it is false again when
  one of them is false.
- An input whose name negates (`not`, `no`, `non`, `never`, `without`, ...)
  does not stand for an item.
- For an item with one `or`, the inputs name both sides.
- No item's inputs are contained in another item's.

Naming an input is not enough: the earlier registry design with a group gate
(#2216, review round 4) was satisfied by `... or L or (A and False)`. Here `A`
in a dead arm changes no value of the formula.

When the formula is outside the closed reader (a rule or an input that is not
Boolean, a comparison, a function call, more than 10 inputs), the clause keeps
the flat reading.

### 3. Binding by structure

The flat reader sees a list only through an excerpt of its chapeau. With the
switch on, a rule is read against the whole list when:

- the chapeau puts the items under its own cue as joint conditions by a closed
  form (`when|if|provided that` + `all | each | both | every one | all three`
  [`of`] `the following ...:`; `if|when the following conditions|requirements|
  criteria are met|apply:`; `when|if <1-4 words> meets|satisfies all|each of
  the following ...:`; a bare `if:`), or has the cue form of alternatives;
- no rule quotes the chapeau;
- a rule quotes two items or more, or uses rules that together quote two.

Then one such rule must pass the reading of the chapeau clause. This only adds
findings. It does not bind: one rule for each item with no combining rule; a
rule that quotes a single item; a count list ("at least 2 of the following");
a consequence list ("When X, all of the following apply:"); a chapeau without
a cue ("All of the following must be met:"), where the flat reader has no
facts either.

## Invariants

| | Invariant | Tests |
|---|---|---|
| (a) | Fail closed: without the closed evidence every clause is read flat, text for text. | property test `test_a_text_without_a_closed_chapeau_form_is_read_exactly_as_flat`; the `..._keeps_the_flat_reading` tests (interrupted lists, reversed chapeaus, qualified items, text after the list, unnamed item words, non-Boolean inputs) |
| (b) | A relaxed list is verified on the value of the formula. Stated limits below. | `test_a_wrong_encoding_of_the_alternatives_is_refused`, `test_a_formula_that_is_not_a_plain_or_is_refused`, `test_rulespec_boolean_literals_are_read_as_constants`, `test_an_alternative_must_hold_in_every_version_of_a_dependency`, `test_an_item_with_one_or_needs_inputs_that_name_both_cases`, `test_one_input_does_not_stand_for_two_facts_of_an_item` |
| (c) | A rule that quotes two items or more is read against the list (closed forms). Narrowed, not closed. | `test_a_dropped_conjunct_is_refused_when_only_items_are_quoted`, `test_a_combining_rule_that_drops_an_item_rule_is_refused` |
| (d) | A conjunctive or unrecognised list is refused at least as often as on the flat reading. | `test_a_conjunctive_list_is_refused_at_least_as_often_as_on_the_flat_reading` |

## Reviewed alternative lists

The registry (`docs/reviewed-alternative-lists.md`) is unchanged and stays the
stronger exemption. A unit whose exact text has a registry entry is not read
structurally at all. The same holds for a reviewed care source, a bound
schedule and a physical-allowance list.

## Limits, stated plainly

What a verified list does not check:

- **The sense of the output.** "A card is withheld when any of the following
  occur" is affirmative grammar. A rule named for the opposite outcome with
  `a or b or c` passes. The reversing-word list catches the wordings it holds
  and no others. The flat reader does not check this either; it refuses these
  candidates for another reason.
- **Names.** An input is matched to a fact by the words of its name, as
  everywhere in this check. Two inputs with the same words, or an input whose
  name says something its values do not, pass.
- **Qualifiers that leave no word.** The word-cover rule sees a qualifier only
  through a word that no input names.
- **A sentence before the list** that states a general condition.

What binding does not reach (the flat reader gives 0 findings in each case):
a rule with no proof excerpt; an excerpt that stands at two places in the text;
a rule that quotes one item; a decoy rule of a type the check does not read
that quotes a conjunctive chapeau; a wrapper `r2 = r and false`; more than 8
lists or 6 bound rules.

Source forms that stay refused as on the flat reading:

- a list of alternatives followed by any text, or with a second condition in
  its chapeau, or with an item that has its own condition or qualifier;
- a `NOTE` inside a list whose chapeau is not one of the two forms;
- a fronted condition that is not ended before its main clause;
- other wordings of alternatives ("may receive any of the following", "either
  situation occurs");
- a list under a negative output;
- lists with labels (`(a)`, `1.`) and nesting by indent.
