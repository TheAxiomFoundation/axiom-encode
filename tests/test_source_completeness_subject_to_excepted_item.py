"""A ``subject to`` that describes an item an ``other than`` excepts.

In "Each registrar shall file every record other than the owner's deed and
the parcel map subject to the lease.", the phrase names the map the list
excepts; it sets no condition on the duty, so the complete-source check
must not demand a case pair for it. A ``subject to``
whose phrase may state a condition on the rule keeps its cue, and so does one
whose place in the sentence the check can't pin down. The sentences here are
constructed or come from public sources.
"""

from __future__ import annotations

import re
from functools import partial

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

# Constructed citations.
CITATION = "us/regulation/99/9999/6"
STATUTE_CITATION = "us/statute/12/9999"

# A constructed paragraph whose second sentence ends in an excepted list.
STALL_SOURCE = (
    "(c) Stall records. Each vendor must display its stall number at the front "
    "of the stall. The market manager may not ask a vendor for any record other "
    "than the vendor's tax number, a sales log, or a floor map of the stall "
    "subject to the lease.\n"
)
# The same paragraph with "other than" replaced, so the phrase no longer sits
# in an excepted list.
STALL_CONTROL = STALL_SOURCE.replace(
    "any record other than the vendor's", "any record and the vendor's"
)
STALL_INPUTS = (
    "stall_number_displayed",
    "vendor_asked_for_other_record",
)

SKIPPED = {
    "excepted list after a prohibition": (
        "The market manager may not ask a vendor for any record other than the "
        "vendor's tax number, a sales log, or a floor map of the stall subject to "
        "the lease."
    ),
    "42 CFR 457.1150(b)": (
        "External review must be conducted by the State or a contractor other "
        "than the contractor responsible for the matter subject to external "
        "review."
    ),
    "enclosed serial list with a cross-reference in the phrase": (
        "Each tenant (other than a farm stand, a food truck, or a craft booth "
        "subject to section 4(b) of the market rules) pays the yearly stall rent."
    ),
    "phrase ends at the enclosing parenthesis": (
        "Any person (other than a bank subject to paragraph (b)) shall file a report."
    ),
    "comma inside a parenthesis in the enclosed phrase": (
        "Any person (other than a bank subject to the rules (as amended, "
        "restated)) shall file a report."
    ),
    "enclosed exception needs no verb before it": (
        "Each person (other than a bank subject to paragraph (b)) files a report."
    ),
    "the nearest other than counts": (
        "Each registrar shall waive every fee other than costs, and shall file "
        "every record other than the parcel map subject to the lease."
    ),
    "four words in the phrase": (
        "Each registrar shall file every record other than the parcel map subject "
        "to the owner's two-year lease."
    ),
    "noun ending in -ed": (
        "The clerk may not record any instrument other than a deed subject to the "
        "owner's request."
    ),
    "line breaks inside the sentence": (
        "Each registrar shall file every record other than the\n"
        "owner's lot number and the map\nsubject to the\n"
        "two-year lease."
    ),
    "pronoun that stands for the excepted item": (
        "The Coast Guard may not inspect any vessel other than one subject to review."
    ),
    "serial list": (
        "The agency may not charge any fee other than copies, records, or a fee "
        "subject to review."
    ),
    # Only a comma before a word of _EXCEPTED_ITEM_PHRASE_LEADS fails the list.
    "comma before a word that leads no phrase": (
        "The agency may not charge any fee other than copies, notwithstanding "
        "section 5, or a fee subject to review."
    ),
    "line break after the opening parenthesis": (
        "Any person (\nother than a bank subject to paragraph (b)) shall file a report."
    ),
    "line break inside other than": (
        "Each registrar shall file every record other\nthan the parcel map "
        "subject to the lease."
    ),
    "question mark ends the phrase": (
        "The agency may not charge any fee other than a copy fee subject to "
        "review? The fee applies to each copy."
    ),
    "exclamation mark ends the phrase": (
        "The agency may not charge any fee other than a copy fee subject to "
        "review! The fee applies to each copy."
    ),
    "verb before a closed parenthesis": (
        "The agency may (in an emergency) impose any fee other than a copy fee "
        "subject to review."
    ),
    # A quotation that opens and closes in the complement, one that closes
    # before the "other than", and a possessive apostrophe leave the
    # exception as it is.
    "quoted term inside the complement": (
        "Each registrar shall file every record other than the “two-year” map "
        "subject to the lease."
    ),
    "quotation closed before other than": (
        "Under the “fee rule” the agency may not charge any fee other than a "
        "copy fee subject to review."
    ),
    "plural possessive in the complement": (
        "Each registrar shall file every record other than the owners’ lot "
        "numbers subject to the lease."
    ),
    "plural possessive with a straight apostrophe in the complement": (
        "Each registrar shall file every record other than the owners' lot "
        "numbers subject to the lease."
    ),
}

# The supervision chapeau, cut by structure (a dash, then subparagraphs) and by
# a colon. The full text keeps the cue, so each clause must keep it too.
CHAPEAUS = {
    "dash": (
        "(c) Reports\n(1) In general\nAny lender other than an insured depository "
        "institution subject to the supervision of the Bureau—\n(A) shall file a "
        "report with the Bureau; and\n(B) shall make the report available to the "
        "public.\n"
    ),
    "colon": (
        "(c) Reports. Any lender other than an insured depository institution "
        "subject to the supervision of the Bureau shall:\n(1) File a report with "
        "the Bureau; and\n(2) Make the report available to the public.\n"
    ),
}

KEPT = {
    # A comma closes the complement before the cue, and "after the end of"
    # attaches to "continue".
    "12 U.S.C. 1467a(c)": (
        "or (C) continue any business activity, other than the activities "
        "described in paragraph (2), after the end of the 2-year period beginning "
        "on the date on which such company received approval under subsection (e) "
        "of this section to become a savings and loan holding company subject to "
        "the limitations contained in this subparagraph."
    ),
    "12 U.S.C. 1467a(c) with following": (
        "or (C) continue any business activity, other than the activities "
        "described in paragraph (2), following the end of the 2-year period "
        "beginning on the date on which such company received approval under "
        "subsection (e) of this section to become a savings and loan holding "
        "company subject to the limitations contained in this subparagraph."
    ),
    "12 U.S.C. 1467a(c) with beyond": (
        "or (C) continue any business activity, other than the activities "
        "described in paragraph (2), beyond the end of the 2-year period beginning "
        "on the date on which such company received approval under subsection (e) "
        "of this section to become a savings and loan holding company subject to "
        "the limitations contained in this subparagraph."
    ),
    # The parenthesis closes before the cue.
    "42 U.S.C. 12709(a)": (
        "new construction of single family housing (other than manufactured "
        "homes) subject to mortgages insured, guaranteed, or made by the Secretary "
        "of Agriculture under title V of the Housing Act of 1949 [ 42 U.S.C. 1471 "
        "et seq.];"
    ),
    # No "other than": the providers § 54.318 covers bear the duty.
    "47 CFR 54.316(a)(10)": (
        "Mobile providers subject to the requirements of § 54.318 shall submit "
        "backhaul and middle mile maps covering eligible areas."
    ),
    "plain condition": "Owners file reports subject to review.",
    "47 CFR 54.308(a)(1)": (
        "Carriers that have elected to receive Connect America Fund-Alternative "
        "Connect America Cost Model (CAF-ACAM) support pursuant to § 54.311, other "
        "than Enhanced A-CAM support, are required to offer broadband service at "
        "actual speeds of at least 10 Mbps downstream/1 Mbps upstream to a defined "
        "number of locations as specified by public notice, with a minimum usage "
        "allowance of 150 GB per month, subject to the requirement that usage "
        "allowances remain consistent with mean usage in the United States over "
        "the course of the term."
    ),
    "12 U.S.C. 1821(d)(14)": (
        "and (ii) in the case of any tort claim (other than a claim which is "
        "subject to section 1441a(b)(14) 2 of this title), the longer of"
    ),
    "7 U.S.C. 2024(f)(2)": (
        "All property, real and personal, used in a transaction or attempted "
        "transaction, to commit, or to facilitate the commission of, a violation "
        "(other than a misdemeanor) of subsection (b) or (c), or proceeds "
        "traceable to a violation of subsection (b) or (c), shall be subject to "
        "forfeiture to the United States under paragraph (1)."
    ),
    "phrase runs into a predicate": (
        "A lender other than a bank subject to paragraph (b) shall file a report."
    ),
    "text ends before the predicate": (
        "A lender other than a bank subject to paragraph (b)"
    ),
    "text ends mid-sentence": (
        "The agency may not charge for any service other than a copy subject to "
        "the fee schedule"
    ),
    "text ends at an abbreviation": (
        "Any order other than an order subject to review by the U.S."
    ),
    "abbreviation in the phrase: U.S. Court": (
        "Any order other than an order subject to review by the U.S. Court of "
        "Appeals for the Federal Circuit shall become final 60 days after issuance."
    ),
    "abbreviation in the phrase: U.S. Supreme Court": (
        "A lender other than a bank subject to review by the U.S. Supreme Court, as "
        "provided in section 1254, shall file a report."
    ),
    "abbreviation in the phrase: Sec. Sec.": (
        "Each State other than a State subject to the requirements of Sec. Sec. "
        "435.550 through 435.563: (1) Enrollment totals."
    ),
    "abbreviation in the phrase: U.S. Department": (
        "A lender other than a bank subject to the U.S. Department of "
        "Agriculture's rules shall file a report."
    ),
    "abbreviation in the phrase: Pub. L.": (
        "A lender other than a bank subject to Pub. L. 111-203 shall file a report."
    ),
    "abbreviation in the phrase: U.S.C. App.": (
        "An agency other than a component subject to 5 U.S.C. App. 2 must publish "
        "a notice."
    ),
    "abbreviation in the phrase: Inc.": (
        "No affiliate other than a subsidiary subject to rules of Acme Inc. "
        "Holdings shall vote."
    ),
    # A capitalized word, or a word after an apostrophe, before a period ends
    # no sentence, so the cue stays.
    "abbreviation in the phrase: Cal. Civ. Code": (
        "Any business other than a covered entity subject to Cal. Civ. Code § "
        "1798.100 shall disclose the data."
    ),
    "abbreviation in the phrase: Ill. Admin. Code": (
        "Any order other than an order subject to Ill. Admin. Code tit. 83 shall "
        "be final."
    ),
    "abbreviation in the phrase: N.Y. Gen. Bus. Law": (
        "A seller other than a dealer subject to N.Y. Gen. Bus. Law art. 29-H "
        "shall register."
    ),
    "abbreviation in the phrase: Dep't.": (
        "A lender other than a bank subject to the Dep't. Of Labor rules shall file."
    ),
    "abbreviation in the phrase: Fed. Res. Bd.": (
        "A lender other than a bank subject to Fed. Res. Bd. Regulation Z shall "
        "file a report."
    ),
    "abbreviation in the phrase: Ex. Order": (
        "A filer other than a person subject to Ex. Order 12866 must report."
    ),
    "abbreviation in the phrase: LLC.": (
        "A partner other than a member subject to Acme LLC. Agreement terms shall vote."
    ),
    "abbreviation in the phrase: Ariz. Rev. Stat.": (
        "A dealer other than a seller subject to Ariz. Rev. Stat. 44-1522 shall "
        "register."
    ),
    "capitalized word ends the phrase": (
        "The agency may not charge any fee other than a copy fee subject to MRRB."
    ),
    "number ends the phrase": (
        "The agency may not charge any fee other than a copy fee subject to form 7."
    ),
    "authority in the phrase": (
        "A lender other than a bank subject to the Department of Agriculture's "
        "rules shall file a report."
    ),
    "condition word in the phrase": (
        "No fee may be charged for any loan other than a loan insured under 12 "
        "U.S.C. 1709 by the Secretary subject to the limits in paragraph (c)."
    ),
    "capitalized condition word in the phrase": (
        "The Secretary shall pay benefits to persons other than veterans subject "
        "to Secretary review."
    ),
    "cross-reference in the phrase": (
        "The agency may not charge for any record other than copies, or for any "
        "service other than a copy subject to paragraph (b)."
    ),
    "long list, cross-reference in the phrase": (
        "No fee may be imposed for any service other than "
        + ", ".join(f"item {letter}" for letter in "abcdefghijklmnopqrst")
        + ", or a copy subject to paragraph (b)."
    ),
    # The phrase may attach to the main clause's verb.
    "phrase may attach to the verb: availability of appropriations": (
        "The Secretary shall make grants to States other than Alaska subject to "
        "the availability of appropriations."
    ),
    "phrase may attach to the verb: approval of the Secretary": (
        "A State may use funds for purposes other than administration subject to "
        "the approval of the Secretary."
    ),
    "phrase may attach to the verb: written consent": (
        "The agency may disclose a record to persons other than the subject "
        "individual subject to the written consent of the individual."
    ),
    "phrase may attach to the verb: income limits": (
        "Benefits shall be paid to individuals other than veterans subject to the "
        "income limits in section 1521."
    ),
    "phrase may attach to the verb: appeal": (
        "The Secretary may pay benefits to persons other than veterans subject to "
        "appeal."
    ),
    "phrase may attach to the verb: notice and hearing": (
        "The agency may revoke licenses other than temporary licenses subject to "
        "notice and hearing."
    ),
    "phrase may attach to the verb: a hearing": (
        "The agency may revoke any license other than a temporary permit subject "
        "to a hearing."
    ),
    "heading cross-reference in the phrase": (
        "Heading 9903.88.03 covers products of China other than products subject "
        "to heading 9903.88.01."
    ),
    "lexical verb in the complement": (
        "A lender other than a bank files reports subject to paragraph (b)."
    ),
    "relative clause in the phrase": (
        "every record other than the owner's permit and the map subject to the "
        "lease which the owner signed."
    ),
    # No modal or finite verb comes before the "other than" in its clause, so
    # the phrase may hold the main clause's verb.
    "no verb before other than: files": (
        "A lender other than a bank subject to review files reports."
    ),
    "no verb before other than: receives": (
        "A State other than a territory subject to review receives a grant."
    ),
    "no verb before other than: applies": (
        "A fee other than a copy fee subject to review applies."
    ),
    "no verb before other than: paid": (
        "A lender other than a bank subject to review paid the fee."
    ),
    "no verb before other than: lexical verb in the complement": (
        "A lender other than a bank files reports subject to review."
    ),
    "no verb before other than: past tense in the complement": (
        "Each lender other than a bank received grants subject to review."
    ),
    "no verb before other than: stays": (
        "Each lender other than a bank stays subject to review."
    ),
    "no verb before other than: seems": (
        "Any loan other than a mortgage seems subject to review."
    ),
    "no verb before other than: stands": (
        "Each loan other than a mortgage stands subject to review."
    ),
    "no verb before other than: falls": (
        "Any fee other than a copy fee falls subject to review."
    ),
    "no verb before other than: label before it": (
        "(b) other than in an emergency the agency imposes fees subject to review."
    ),
    "no verb before other than: the excepted item is the subject": (
        "A household other than a household subject to MRRB reports changes."
    ),
    "verb inside a closed parenthesis before other than": (
        "The agency (which may charge fees) imposes any fee other than a copy fee "
        "subject to review."
    ),
    "verb before a parenthesis still open at other than": (
        "The agency may impose fees (for any fee other than a copy fee subject to "
        "review."
    ),
    # A comma, a dash or a capital before "other than" leaves the
    # complement's end unknown.
    "fronted complement ends at its comma": (
        "Other than as provided in paragraph (b), this section applies to fees "
        "and charges subject to paragraph (c)."
    ),
    "modal after the complement": (
        "Other than in an emergency, the agency may impose a fee subject to the "
        "limits in paragraph (c)."
    ),
    "capitalized Other than with a list": (
        "Other than copies, records, or files subject to paragraph (b), the agency "
        "may not charge."
    ),
    "other than opens the text": (
        "other than a farm stand, a food truck or a craft booth subject to "
        "section 4(b) of the market rules."
    ),
    "comma before other than: grants or contracts": (
        "The Secretary may, other than as provided in subsection (b), make grants "
        "or contracts subject to the availability of appropriations."
    ),
    "comma before other than: grants": (
        "The Secretary may, other than as provided in subsection (b), make grants "
        "subject to the availability of appropriations."
    ),
    "comma before other than: penalties and fees": (
        "The Administrator shall, other than in the case of a small entity, impose "
        "penalties and fees subject to the limits of paragraph (d)."
    ),
    "comma before other than: fee or charge": (
        "The agency may not, other than in an emergency, impose any fee or charge "
        "subject to the limits in paragraph (c)."
    ),
    "comma before other than: in addition": (
        "In addition, other than for emergencies, fees and charges subject to "
        "paragraph (b) apply."
    ),
    "comma before other than: grants and loans": (
        "Each State, other than Alaska, receives grants and loans subject to the "
        "availability of appropriations."
    ),
    "dash before other than": (
        "The agency may not charge any fee — other than a copy fee subject to review."
    ),
    "hyphen before other than": (
        "The agency may not charge any fee - other than a copy fee subject to review."
    ),
    "comma before the cue": (
        "The agency may not charge for any service other than copies, subject to "
        "paragraph (b)."
    ),
    "comma ends the phrase": (
        "The agency may not charge for any service other than a copy subject to "
        "paragraph (b), in any year."
    ),
    "semicolon ends the phrase": (
        "The agency may not charge for any service other than a copy subject to "
        "paragraph (b); or for any record."
    ),
    "stray closing parenthesis": (
        "The agency may not charge for any service other than a copy subject to "
        "item 2) of the schedule."
    ),
    "relative clause stays on the safe side": (
        "The agency may not charge for any service other than copies that remain "
        "subject to paragraph (b)."
    ),
    "relative clause with a lexical verb": (
        "The agency may not charge for any service other than copies that the "
        "agency holds subject to paragraph (b)."
    ),
    "semicolon between": (
        "The agency may not charge for any service other than copies; a copy fee "
        "subject to paragraph (b)."
    ),
    "em dash between": (
        "The agency may not charge any fee other than a copy fee — the agency's "
        "records subject to paragraph (b)."
    ),
    "two hyphens between": (
        "The agency may not charge any fee other than a copy fee -- the agency's "
        "records subject to paragraph (b)."
    ),
    "predicate before the enclosing parenthesis": (
        "Any person (other than a bank subject to rules that are in paragraph (b)) "
        "shall file a report."
    ),
    "period in the enclosed phrase": (
        "Any person (other than a bank subject to 12 U.S.C. 1841) shall file a report."
    ),
    # A clause that the semicolon cuts off from its verb.
    "nearest other than after a semicolon": (
        "Each registrar shall waive every fee other than costs; or every record "
        "other than the parcel map subject to the lease."
    ),
    # A period or a capital in the complement keeps the cue: the check can't
    # tell an abbreviation inside the list from the end of a sentence.
    "abbreviation inside the complement": (
        "Each registrar shall file every record other than the owner's U.S. "
        "Postal Service address and the parcel map subject to the lease."
    ),
    "complement crosses a sentence after U.S.": (
        "Fees apply to vessels other than vessels built in the U.S. Owners file "
        "reports subject to review."
    ),
    "complement crosses a sentence after Inc.": (
        "Fees apply to vessels other than vessels built by Acme Inc. Owners file "
        "reports subject to review."
    ),
    "complement crosses a sentence after et seq.": (
        "Fees apply to loans other than loans under 42 U.S.C. 1471 et seq. Owners "
        "file reports subject to review."
    ),
    "complement crosses a sentence after an abbreviation into lower case": (
        "The rule applies to any loan other than a loan made by Acme Inc. Interest "
        "on a loan subject to review accrues daily."
    ),
    "sentence ends between": (
        "The agency may not charge for any service other than copies. A copy fee "
        "subject to paragraph (b) applies."
    ),
    "label after a sentence end between": (
        "The agency may not charge for any service other than copies. (b) A copy "
        "fee subject to paragraph (c) applies."
    ),
    "digit starts the next sentence": (
        "The agency may not charge for any service other than copies. 2 CFR part "
        "200 governs each grant subject to review."
    ),
    "section sign starts the next sentence": (
        "The agency may not charge for any service other than copies. § 2 governs "
        "each grant subject to review."
    ),
    "lowercase word starts the next sentence": (
        "The agency may not charge for any service other than copies. eFile users "
        "pay a fee subject to review."
    ),
    "single quote starts the next sentence": (
        "The agency may not charge for any service other than copies. ‘Covered "
        "fees’ include a fee subject to review."
    ),
    "capital in the complement": (
        "The agency may not charge any fee other than the Secretary's copy fee "
        "subject to review."
    ),
    "accented word right before the cue": (
        "The agency may not charge any fee other than Écoles subject to review."
    ),
    "accented capital in the complement": (
        "The agency may not charge any fee other than École fees subject to review."
    ),
    "heading without a period": (
        "(2) Entities other than States\nEach entity subject to subsection (a) "
        "receives a grant under paragraph (3)."
    ),
    "heading without a period, then a modal": (
        "(2) Entities other than States\nEach entity subject to subsection (a) "
        "shall receive a grant under paragraph (3)."
    ),
    "heading without a period, short phrase": (
        "(2) Entities other than States\nEach entity subject to review receives a "
        "grant."
    ),
    "bullet lines": "Income other than\n• wages\n• salaries subject to review.",
    "lowercase label line": (
        "The agency may not charge for any service other than copies\na. each "
        "grant subject to review accrues."
    ),
    "et seq. before the cue": (
        "A loan other than a loan made under 42 U.S.C. 1471 et seq. subject to "
        "paragraph (b) shall be reported."
    ),
    "nearest other than inside a closed parenthesis": (
        "Any loan other than a loan (other than a mortgage) subject to paragraph "
        "(b) is exempt."
    ),
    "closing quote before the cue": (
        "No fee other than a “processing fee” subject to paragraph (b) may be charged."
    ),
    "closing quote, then the end of the sentence": (
        "No person may charge any fee other than the “processing fee” subject to "
        "paragraph (b)."
    ),
    "cross-reference before the cue": (
        "The agency may not charge for any service other than a copy under section "
        "5(b) subject to paragraph (c)."
    ),
    "finite verb in the complement": (
        "The agency may not charge for any service other than a copy that is "
        "requested subject to paragraph (b)."
    ),
    # Modals and finite verbs an earlier version of the check missed. A verb
    # before the "other than" leaves each of these with one reason to keep.
    "cannot in the complement": (
        "The agency may provide that a fee other than a copy fee cannot rise "
        "subject to review."
    ),
    "cannot in the phrase": (
        "The agency may require that a lender other than a bank subject to review "
        "cannot file."
    ),
    "cannot with no verb before other than": (
        "A lender other than a bank cannot file reports subject to review."
    ),
    "need not in the complement": (
        "The agency may provide that a creditor other than a card issuer need not "
        "provide refunds subject to review."
    ),
    "ought in the complement": (
        "The agency may provide that a lender other than a bank ought to file "
        "reports subject to review."
    ),
    "contraction in the complement: isn't": (
        "The agency may provide that any fee other than a copy fee isn't payable "
        "subject to review."
    ),
    "contraction in the complement: doesn't": (
        "The agency may provide that a lender other than a bank doesn't file "
        "reports subject to review."
    ),
    "contraction in the complement: won’t": (
        "The agency may provide that a lender other than a bank won’t receive "
        "grants subject to review."
    ),
    "contraction in the complement: aren't": (
        "The agency may provide that fees other than copy fees aren't refundable "
        "subject to review."
    ),
    "contraction in the phrase": (
        "The agency may require that a lender other than a bank subject to review "
        "isn't filing."
    ),
    # A quantifier after a conjunction may open a second object of the verb.
    "quantified sibling: or any": (
        "The agency may not impose any fee other than a copy fee or any charge "
        "subject to review."
    ),
    "quantified sibling: and each": (
        "The agency may not impose any fee other than a copy fee and each penalty "
        "subject to review."
    ),
    "quantified sibling: nor any": (
        "No person may charge any fee other than a copy fee nor any penalty "
        "subject to review."
    ),
    "list without a closing conjunction": (
        "The agency may not charge for any service other than copies, the "
        "agency's records subject to paragraph (b)."
    ),
    "adverb before the cue": (
        "The agency may not charge for any service other than one provided "
        "directly subject to paragraph (b)."
    ),
    "reflexive pronoun before the cue": (
        "No person may assign the claim to any person other than itself subject to "
        "paragraph (b)."
    ),
    "object pronoun before the cue": (
        "The officer may not delegate the duty to any person other than him "
        "subject to paragraph (b)."
    ),
    "noun ending in -ly stays on the safe side": (
        "The agency may not charge any household other than a family subject to "
        "paragraph (b)."
    ),
    "digit runs into the word before the cue": (
        "The agency may not charge any fee other than a 401k subject to review."
    ),
    "capitalized cue after a line break": (
        "Each registrar shall file every record other than the parcel "
        "map\nSubject to the lease."
    ),
    "parenthesis opened in the complement": (
        "The agency may not charge for any service other than copies (including "
        "any copy subject to paragraph (b)."
    ),
    "cue opens its sentence": (
        "The agency may not charge for any service other than copies. Subject to "
        "paragraph (b), the agency may waive the fee."
    ),
    "beyond the window": (
        "The agency may not charge for any service other than the "
        + "large " * 70
        + "copy subject to paragraph (b)."
    ),
    # Quoted, printed or inserted text, whose "other than" may except
    # nothing: the excepted item starts inside a quotation or square bracket,
    # or crosses into or out of one.
    "other than inside curly double quotes": (
        "We may print “other than loans” on forms subject to review."
    ),
    "other than inside curly single quotes": (
        "We may print ‘other than loans’ on forms subject to review."
    ),
    "other than inside square brackets": (
        "We may print [other than loans] on forms subject to review."
    ),
    "other than inside straight double quotes": (
        'We may print "other than loans" on forms subject to review.'
    ),
    "other than inside straight single quotes": (
        "We may print 'other than loans' on forms subject to review."
    ),
    "quotation opened earlier in the sentence closes in the complement": (
        "We may print “fees other than loans” on forms subject to review."
    ),
    "cue inside a quotation the complement opens": (
        "The agency may not charge any fee other than a “copy fee subject to review.”"
    ),
    "exception inside quoted amendatory text": (
        "Section 5 is amended by inserting “The agency may not charge any fee "
        "other than a copy fee subject to review.”"
    ),
    "quotation closes only after the sentence end": (
        "The agency may not charge any fee other than a copy fee subject to "
        "review.” Fees apply."
    ),
    **{f"chapeau ending at a {name}": text for name, text in CHAPEAUS.items()},
}

# Shapes the check still drops although the phrase may state a condition on
# the rule. Each one matches a line of the PR's Known limits.
KNOWN_LIMIT_DROPS = {
    "verb before other than, short phrase at the end": (
        "The Secretary shall pay benefits to persons other than veterans subject "
        "to review."
    ),
    "verb before other than, lexical verb in the complement": (
        "The agency may require that a lender other than a bank files reports "
        "subject to review."
    ),
    "verb in a relative clause before other than": (
        "A lender that is a bank other than a credit union subject to review files "
        "reports."
    ),
    "verb in a subordinate clause before other than": (
        "If the agency may act, fees other than copies subject to review apply."
    ),
    "verb before other than, lexical verb after the phrase": (
        "The agency may require that a lender other than a bank subject to review "
        "files reports."
    ),
    "lowercase line without a mark before it": (
        "The agency may not charge for any service other than copies\neach grant "
        "subject to review."
    ),
}

# One sentence whose cue drops, and one-change variants of it that each keep
# the cue for a single reason.
BASE = (
    "Each registrar shall file every record other than the owner's deed and the "
    "parcel map subject to the lease."
)
MINIMAL_PAIRS = {
    "no other than": ("record other than", "record and"),
    "nothing before other than": (
        "Each registrar shall file every record other than",
        "other than",
    ),
    "no verb before other than": ("shall file", "files"),
    "verb inside a closed parenthesis before other than": (
        "Each registrar shall file",
        "Each registrar (who may act) files",
    ),
    "verb before a parenthesis still open at other than": (
        "file every",
        "file (every",
    ),
    "semicolon between the verb and other than": ("file every", "file; every"),
    "colon between the verb and other than": ("file every", "file: every"),
    "period between the verb and other than": ("file every", "file. every"),
    "exclamation mark between the verb and other than": (
        "file every",
        "file! every",
    ),
    "question mark between the verb and other than": ("file every", "file? every"),
    "em dash between the verb and other than": ("file every", "file — every"),
    "en dash between the verb and other than": ("file every", "file – every"),
    "two hyphens between the verb and other than": ("file every", "file -- every"),
    "new paragraph between the verb and other than": (
        "file every",
        "file\n(b) every",
    ),
    "capitalized cue": ("subject to", "Subject to"),
    "capitalized to in the cue": ("subject to", "subject To"),
    "reflexive pronoun before the cue": ("the parcel map subject", "itself subject"),
    "word ending in -ly before the cue": (
        "map subject",
        "map directly subject",
    ),
    "word ending in -ed before the cue": ("map subject", "map listed subject"),
    "closing quote before the cue": (
        "the parcel map subject",
        "the “parcel map” subject",
    ),
    "opening quote before other than": ("record other", "record “other"),
    "opening single quote before other than": (
        "record other",
        "record ‘other",
    ),
    "opening square bracket before other than": (
        "record other",
        "record [other",
    ),
    "closing quote in the complement": ("deed and", "deed” and"),
    "closing square bracket in the complement": ("deed and", "deed] and"),
    "quotation the complement opens holds the cue": (
        "the parcel map",
        "the “parcel map",
    ),
    "closing quote after the sentence end": (
        "lease.",
        "lease.”",
    ),
    "capital outside Latin-1 in the complement": (
        "the owner's deed",
        "the owner's Łódź deed",
    ),
    "digit runs into the word before the cue": (
        "the parcel map subject",
        "the parcel 401k subject",
    ),
    "digit and hyphen before the word before the cue": (
        "the parcel map subject",
        "the parcel 401-unit subject",
    ),
    "hyphen joins the word to the cue": ("map subject", "map-subject"),
    "capitalized Other than": ("other than", "Other than"),
    "comma before other than": ("record other", "record, other"),
    "comma and a line break before other than": (
        "record other",
        "record,\nother",
    ),
    "hyphen before other than": ("record other", "record - other"),
    "semicolon before other than": ("record other", "record; other"),
    "colon before other than": ("record other", "record: other"),
    "period before other than": ("record other", "record. other"),
    "period with no space before other than": (
        "record other",
        "record.other",
    ),
    "em dash before other than": ("record other", "record — other"),
    "en dash before other than": ("record other", "record – other"),
    "two hyphens before other than": ("record other", "record -- other"),
    "exclamation mark before other than": ("record other", "record! other"),
    "question mark before other than": ("record other", "record? other"),
    "semicolon in the complement": ("deed and", "deed; and"),
    "colon in the complement": ("deed and", "deed: and"),
    "em dash in the complement": ("deed and", "deed — and"),
    "en dash in the complement": ("deed and", "deed – and"),
    "two hyphens in the complement": ("deed and", "deed -- and"),
    "sentence ends in the complement": ("deed and the", "deed. And the"),
    "period before a lowercase word in the complement": (
        "deed and the",
        "deed. and the",
    ),
    "question mark in the complement": ("deed and the", "deed? and the"),
    "exclamation mark in the complement": ("deed and the", "deed! and the"),
    "capital in the complement": ("the owner's deed", "the owner's Deed"),
    "label after a sentence end in the complement": ("deed and the", "deed. (b) The"),
    "closing quote after a sentence end in the complement": (
        "deed and the",
        'deed." The',
    ),
    "opening quote after a sentence end in the complement": (
        "deed and the",
        'deed. "The',
    ),
    "new paragraph in the complement": ("deed and", "deed\n(b) and"),
    "new section in the complement": ("deed and", "deed\n§ 2 and"),
    "digit line in the complement": ("deed and", "deed\n2 and"),
    "bullet line in the complement": ("deed and", "deed\n• and"),
    "blank line in the complement": ("deed and", "deed\n\nand"),
    "modal in the complement": (
        "the parcel map",
        "the map the owner has on file",
    ),
    "contraction in the complement": (
        "the parcel map",
        "the map the owner isn't using",
    ),
    "clause word in the complement": ("the parcel map", "that parcel map"),
    "quantifier after the conjunction in the complement": (
        "deed and the",
        "deed and any",
    ),
    "parenthesis opened in the complement": ("deed and", "deed (and"),
    "parenthesis closed before it opens in the complement": (
        "deed and the",
        "deed) and (the",
    ),
    "list without a serial conjunction": (
        "deed and the parcel",
        "deed, the lot number and the parcel",
    ),
    "phrase lead after a comma": ("deed and", "deed, after notice, and"),
    "phrase lead after the last conjunction": ("deed and", "deed, and after notice"),
    "phrase lead after a last or": ("deed and", "deed, or after notice"),
    "phrase lead after a last nor": ("deed and", "deed, nor after notice"),
    "comma in the phrase": ("lease.", "lease, once."),
    "semicolon in the phrase": ("lease.", "lease; once."),
    "colon in the phrase": ("lease.", "lease: once."),
    "em dash in the phrase": ("lease.", "lease — once."),
    "en dash in the phrase": ("lease.", "lease – once."),
    "two hyphens in the phrase": ("lease.", "lease -- once."),
    "new paragraph in the phrase": (
        "the lease.",
        "the\n(b) lease.",
    ),
    "lowercase label line in the phrase": (
        "the lease.",
        "the\na. lease.",
    ),
    "modal in the phrase": ("the lease.", "the lease shall apply."),
    "contraction in the phrase": (
        "the lease.",
        "the lease isn't final.",
    ),
    "five words in the phrase": (
        "the lease.",
        "the owner's old ground lease.",
    ),
    "condition word in the phrase": (
        "the lease.",
        "the applicable lease.",
    ),
    "named authority in the phrase": (
        "the lease.",
        "lease by Acme today.",
    ),
    "named authority in capitals in the phrase": (
        "the lease.",
        "lease by FDIC today.",
    ),
    "named authority after the in the phrase": (
        "the lease.",
        "by the Acme now.",
    ),
    "named authority after of in the phrase": (
        "the lease.",
        "forms of Acme now.",
    ),
    "named authority with a capital outside ASCII in the phrase": (
        "the lease.",
        "lease by Ørsted today.",
    ),
    "word with a capital outside ASCII before the period": (
        "the lease.",
        "the Ébc.",
    ),
    "section sign in the phrase": ("the lease.", "§ 5 today."),
    "capitalized word before the period": ("the lease.", "the FDIC."),
    "word after an apostrophe before the period": (
        "the lease.",
        "the Dep't.",
    ),
    "number before the period": ("the lease.", "form 7."),
    "single letter before the period": ("lease.", "lease e."),
    "lowercase abbreviation before the period": (
        "lease.",
        "lease etc.",
    ),
    "capitalized word before a period, then a short sentence": (
        "the lease.",
        "MRRB. Fees apply.",
    ),
    "number before a period, then a short sentence": (
        "the lease.",
        "form 7. Fees apply.",
    ),
    "lowercase abbreviation before a period, then a short sentence": (
        "lease.",
        "lease etc. Fees apply.",
    ),
    "period in a nested parenthesis in the phrase": (
        "the lease.",
        "forms (e.g. 5) here.",
    ),
    "sentence end in a nested parenthesis in the phrase": (
        "the lease.",
        "forms (see here. Fees) now.",
    ),
    "text ends without a sentence end": ("lease.", "lease"),
    "closing parenthesis in the phrase": ("lease.", "lease) here."),
    "parenthesis opened in the phrase": (
        "the lease.",
        "the (lease.",
    ),
}
ENCLOSED_BASE = SKIPPED["phrase ends at the enclosing parenthesis"]
ENCLOSED_PAIRS = {
    "parenthesis never closes": ("(b)) shall", "(b) shall"),
    "sentence ends inside the parenthesis": ("(b))", "(b). The bank)"),
    "period inside the parenthesis": ("paragraph (b))", "12 U.S.C. 1841)"),
    "question mark inside the parenthesis": ("(b))", "(b)? The bank)"),
    "exclamation mark inside the parenthesis": ("(b))", "(b)! The bank)"),
    "semicolon inside the parenthesis": ("(b))", "(b); the bank)"),
    "colon inside the parenthesis": ("(b))", "(b): the bank)"),
    "em dash inside the parenthesis": ("(b))", "(b) — the bank)"),
    "en dash inside the parenthesis": ("(b))", "(b) – the bank)"),
    "two hyphens inside the parenthesis": ("(b))", "(b) -- the bank)"),
    "comma before the closing parenthesis": ("(b))", "(b), as amended)"),
    "predicate before the closing parenthesis": (
        "paragraph (b))",
        "rules that are in paragraph (b))",
    ),
    "lowercase label line before the closing parenthesis": (
        "paragraph (b))",
        "the\na) rules)",
    ),
}


def _apply(text: str, change: tuple[str, str]) -> str:
    old, new = change
    assert text.count(old) == 1, (old, text)
    return text.replace(old, new)


MINIMAL_TEXTS = [_apply(BASE, change) for change in MINIMAL_PAIRS.values()]
ENCLOSED_TEXTS = [_apply(ENCLOSED_BASE, change) for change in ENCLOSED_PAIRS.values()]


def _subject_to_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\bsubject\s+to\b", text, re.I)]


def _kept_subject_to_spans(text: str) -> list[tuple[int, int]]:
    return [
        match.span()
        for match in sc._source_exception_or_applicability_matches(text)
        if re.fullmatch(r"subject\s+to", match.group(0), flags=re.IGNORECASE)
    ]


def _cues(text: str) -> set[tuple[int, int, str]]:
    return {
        (match.start(), match.end(), match.group(0))
        for match in sc._source_exception_or_applicability_matches(text)
    }


def _unfiltered_cues(text: str) -> set[tuple[int, int, str]]:
    """The cues with the excepted-item filter switched off, as on main."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sc, "_subject_to_describes_excepted_item", lambda *_args, **_kw: False
        )
        return _cues(text)


def _decisions(text: str) -> dict[int, bool]:
    """Whether the filter drops each ``subject to`` in ``text``, by position."""

    return {
        match.start(): sc._subject_to_describes_excepted_item(text, match)
        for match in re.finditer(r"\bsubject\s+to\b", text, flags=re.IGNORECASE)
    }


@pytest.mark.parametrize("text", SKIPPED.values(), ids=SKIPPED.keys())
def test_excepted_item_subject_to_states_no_condition(text):
    assert len(_subject_to_spans(text)) == 1
    assert _kept_subject_to_spans(text) == []


@pytest.mark.parametrize("text", KEPT.values(), ids=KEPT.keys())
def test_other_subject_to_cues_keep_their_condition(text):
    assert _kept_subject_to_spans(text) == _subject_to_spans(text)
    assert _subject_to_spans(text)


@pytest.mark.parametrize(
    "text", KNOWN_LIMIT_DROPS.values(), ids=KNOWN_LIMIT_DROPS.keys()
)
def test_known_limits_still_drop_their_cue(text):
    assert list(_decisions(text).values()) == [True]


def test_minimal_pair_bases_drop_their_cue():
    assert _kept_subject_to_spans(BASE) == []
    assert _kept_subject_to_spans(ENCLOSED_BASE) == []


@pytest.mark.parametrize("text", MINIMAL_TEXTS, ids=MINIMAL_PAIRS.keys())
def test_one_change_to_the_sentence_keeps_the_cue(text):
    assert _subject_to_spans(text)
    assert _kept_subject_to_spans(text) == _subject_to_spans(text)


@pytest.mark.parametrize("text", ENCLOSED_TEXTS, ids=ENCLOSED_PAIRS.keys())
def test_one_change_to_the_enclosed_exception_keeps_the_cue(text):
    assert _kept_subject_to_spans(text) == _subject_to_spans(text)


def test_other_cues_after_an_excepted_item_keep_their_condition():
    text = (
        "The agency may not charge for any service other than a copy unless the "
        "applicant consents."
    )
    assert [cue for _start, _end, cue in _cues(text)] == ["unless"]


def _with_complement_padding(distance: int) -> str:
    """BASE's shape with "other than" ``distance`` characters before the cue."""

    head = "Each registrar shall file every record other than"
    tail = "the parcel map subject to the lease."
    padding = distance - len("other than") - len("the parcel map ")
    text = head + " " * padding + tail
    assert text.index("subject to") - text.index("other than") == distance
    return text


def _with_phrase_padding(reach: int) -> str:
    """BASE's shape with the phrase's period ``reach`` characters after the cue."""

    head = "Each registrar shall file every record other than the parcel map subject to"
    tail = "the lease."
    text = head + " " * (reach - len(tail) + 1) + tail
    assert text.index(".", len(head)) - len(head) == reach
    return text


def test_window_bounds_the_other_than_search():
    assert _kept_subject_to_spans(_with_complement_padding(400)) == []
    assert _kept_subject_to_spans(_with_complement_padding(401))
    assert _kept_subject_to_spans(KEPT["beyond the window"])


def test_window_bounds_the_phrase():
    assert _kept_subject_to_spans(_with_phrase_padding(399)) == []
    assert _kept_subject_to_spans(_with_phrase_padding(400))


def test_window_bounds_the_mark_before_other_than():
    # Within the window, the parenthesis encloses the exception; past it, only
    # whitespace lies before the "other than", and the cue stays.
    for gap, drops in ((399, True), (400, False)):
        text = (
            "Any person ("
            + " " * gap
            + "other than a bank subject to paragraph (b)) shall file a report."
        )
        assert list(_decisions(text).values()) == [drops], gap


def test_window_bounds_the_clause_before_other_than():
    for gap, found in ((397, True), (398, False)):
        text = "may" + " " * gap + "other than"
        end = text.index("other than")
        assert sc._excepted_item_clause_has_predicate(text, end) is found, gap


def test_only_the_excepted_item_subject_to_drops_from_the_paragraph():
    unfiltered = _unfiltered_cues(STALL_SOURCE)
    filtered = _cues(STALL_SOURCE)
    assert filtered <= unfiltered
    assert [cue for _start, _end, cue in unfiltered - filtered] == ["subject to"]


@pytest.mark.parametrize(
    ("before", "mark", "after", "ends"),
    [
        ("review by the U.S", ".", " Court of Appeals", False),
        ("subject to Pub", ".", " L. 111-203", False),
        ("subject to Pub. L", ".", " No. 111-203", False),
        ("the requirements of Sec", ".", " Sec. 435.550", False),
        ("rules of Acme Inc", ".", " Holdings shall vote", False),
        ("review by the U.S", ".", "", False),
        ("subject to Cal", ".", " Civ. Code", False),
        ("review by the FDIC", ".", " The agency", False),
        ("subject to the Dep't", ".", " Of Labor", False),
        ("subject to the Dep'ts", ".", " Of Labor", False),
        ("subject to the Dep’ts", ".", " Of Labor", False),
        ("subject to paragraph (b)", ".", " The agency", False),
        ("under 12 U.S.C. 1709", ".", " The agency", False),
        ("the Agency's", ".", " The fee", False),
        ("the request e", ".", " The fee", False),
        ("the request et seq", ".", " The fee", False),
        ("the request etc", ".", " The fee", False),
        ("the lease", ".", " The agency", True),
        ("the lease", ".", "", True),
        ("the lease", ".", "\nand the fee", True),
        ("the lease", ".", " and the fee", False),
        ("the lease", ".", " Ärzte", True),
        ("the lease", ".", " Öl", True),
        ("the lease", ".", " Über", True),
        # A capital in any script starts the next sentence.
        ("the lease", ".", " Élan", True),
        ("the lease", ".", " Łódź", True),
        ("the lease", ".", " Ωmega", True),
        ("the lease", ".", " Жилье", True),
        ("the lease", ".", " ǅemal", True),
        ("the lease", ".", " élan", False),
        # A letter outside ASCII runs into the word before the period.
        ("review at Ébc", ".", " The fee", False),
        ("the Œuvre", ".", " The fee", False),
        ("the naïve", ".", " The fee", False),
        ("the lease", ".", "\tThe fee", True),
        ("the lease", ".", "' The fee", True),
        ("the lease", ".", "’ The fee", True),
        ("the lease", ".", "] The fee", True),
        ("the 5th", ".", " The fee", True),
        ("Is it the request", "?", " The fee", True),
        ("Is it the U.S", "!", " The fee", True),
        ("§ 54", ".", "318", False),
        ("as described in subpar", ".", " (A)", False),
        ("the lease", ".", '" The fee', True),
        ("the lease", ".", "” The fee", True),
        ("the lease", ".", ' "The fee', True),
        ("the lease", ".", " “The fee", True),
        ("the lease", ".", "” and the fee", False),
        ("the lease", ".", " ‘The fee", False),
        ("the lease", ".", " 2 CFR", False),
    ],
)
def test_sentence_ends_only_after_a_sentence_word(before, mark, after, ends):
    text = before + mark + after
    assert sc._excepted_item_sentence_ends_at(text, len(before)) is ends


def test_window_bounds_the_word_before_a_period():
    for letters, ends in ((24, True), (25, False)):
        before = "subject to " + "x" * letters
        text = before + ". The fee"
        assert sc._excepted_item_sentence_ends_at(text, len(before)) is ends, letters


def test_window_bounds_the_quotation_look_back():
    # A quotation opened at most the window's width before the "other than"
    # holds the exception, and the cue stays; one opened further back is out
    # of sight.
    head = "Each registrar shall file every record "
    tail = "other than the parcel map subject to the lease."
    for distance, drops in ((400, False), (401, True)):
        text = "“" + " " * (distance - 1 - len(head)) + head + tail
        assert text.index("other than") - text.index("“") == distance
        assert list(_decisions(text).values()) == [drops], distance


@pytest.mark.parametrize(
    ("text", "crosses"),
    [
        ("other than loans subject to review.", False),
        ("“other than loans” on forms subject to review.", True),
        ("‘other than loans’ on forms subject to review.", True),
        ("[other than loans] on forms subject to review.", True),
        ('"other than loans" on forms subject to review.', True),
        ("'other than loans' on forms subject to review.", True),
        ("«other than loans» on forms subject to review.", True),
        ("“fees other than loans” on forms subject to review.", True),
        ("other than loans” on forms subject to review.", True),
        ("other than loans] on forms subject to review.", True),
        ("other than a “copy fee subject to review.”", True),
        ("other than a copy fee subject to review.”", True),
        ("other than a copy fee subject to the “request.", True),
        ("other than the “processing fee” and copies subject to review.", False),
        ("other than the [listed] copies subject to review.", False),
        ("other than the 'processing fee' and copies subject to review.", False),
        ("other than a copy fee subject to the “request.”", False),
        ("“Fees” apply. Any fee other than copies subject to review.", False),
        ("‘Fees’ apply. Any fee other than copies subject to review.", False),
        ("He wrote “fees. Any fee other than copies subject to review.", True),
        # A mark closes only a quotation of its own kind.
        ("‘Fees” apply. Any fee other than copies subject to review.", True),
        ("[Fees” apply. Any fee other than copies subject to review.", True),
        # Apostrophes and primes.
        ("other than the applicant's name subject to review.", False),
        ("other than the applicant’s name subject to review.", False),
        ("other than the applicants’ names subject to review.", False),
        ("other than the applicants' names subject to review.", False),
        ("other than the Dep’t subject to review.", False),
        ("other than 5' pipes subject to review.", False),
        # A straight double quote that closes nothing counts, even as an inch
        # mark.
        ('other than 12" pipes subject to review.', True),
        # A single closing quote after a space closes something opened before.
        ("other than ’90s models subject to review.", True),
    ],
)
def test_excepted_item_meets_a_quotation_edge(text, crosses):
    start = text.index("other than")
    cue = text.index("subject to")
    assert (
        sc._excepted_item_exception_crosses_quotation(text, start, cue, len(text))
        is crosses
    )


@pytest.mark.parametrize(
    ("text", "role"),
    [
        ("“", "open"),
        ("‘", "open"),
        ("«", "open"),
        ("[", "open"),
        ("”", "close"),
        ("’", "close"),
        ("»", "close"),
        ("]", "close"),
        ('"loan', "open"),
        ('a "loan', "open"),
        ('("loan', "open"),
        ('loan"', "close"),
        ('loan" a', "close"),
        ('a " loan', "close"),
        ("'loan", "open"),
        ("loan'", "close"),
        ("applicant's", ""),
        ("Dep’t", ""),
        ("o‘clock", ""),
    ],
)
def test_quotation_mark_role(text, role):
    index = next(i for i, c in enumerate(text) if c in "\"'“”‘’«»[]")
    assert sc._excepted_item_quotation_mark_role(text, index) == role


@pytest.mark.parametrize(
    ("before", "role"),
    [
        *((mark, "open") for mark in "([{“‘«—–-/"),
        *((mark, "close") for mark in "5).,;”’»]"),
    ],
)
def test_straight_quote_role_follows_the_mark_before_it(before, role):
    # Each opening mark, dash, hyphen or slash before a straight quote that a
    # word follows makes it open; a digit, closing mark or punctuation leaves
    # it closing.
    for quote in "\"'":
        text = f"{before}{quote}loan"
        assert sc._excepted_item_quotation_mark_role(text, 1) == role, text


def test_a_straight_quote_after_a_dash_slash_or_curly_quote_holds_the_exception():
    sentence = (
        '"The agency may not charge any fee other than copies subject to review. '
        'Fees apply."'
    )
    for lead in ("Insert at the end—", "He wrote /", "He wrote “"):
        assert list(_decisions(lead + sentence).values()) == [False], lead


@pytest.mark.parametrize(
    ("phrase", "condition"),
    [
        (" the lease", False),
        (" the applicable request", True),
        (" § 5", True),
        (" review by Acme", True),
        (" review by the Secretary", True),
        (" review by The staff", True),
        (" review by THE staff", True),
        (" review by Ørsted", True),
        (" review of the Ørsted staff", True),
        (" forms of by Ørsted", True),
        (" review by the staff", False),
        (" review by 5 members", False),
        (" review by the 5 members", False),
        (" review thereof Acme", False),
        (" review by Élan", True),
        (" review by élan", False),
    ],
)
def test_phrase_names_a_condition(phrase, condition):
    assert sc._excepted_item_phrase_has_condition(phrase) is condition


@pytest.mark.parametrize(
    "character",
    ["A", "Z", "É", "Ø", "Ł", "Œ", "Ω", "Ж", "ǅ", "ẞ"],
)
def test_capital_in_any_script(character):
    assert sc._excepted_item_is_capital(character)


@pytest.mark.parametrize(
    "character",
    ["a", "é", "ß", "ω", "ж", "5", "(", "“", " ", "_"],
)
def test_not_a_capital(character):
    assert not sc._excepted_item_is_capital(character)


@pytest.mark.parametrize(
    ("complement", "separates"),
    [
        (" a deed, a plat, or a lease ", True),
        (" copies, records, nor files ", True),
        (" copies ", True),
        (" copies, notwithstanding section 5, or a fee ", True),
        (" a deed, a plat or a lease ", False),
        (" copies, after notice, and records ", False),
        (" copies, and after notice records ", False),
        (" copies, the agency's records ", False),
        (" copies, orders ", False),
        (" copies, And records ", True),
        (" copies, AFTER notice, or records ", False),
    ],
)
def test_commas_must_separate_listed_items(complement, separates):
    assert sc._excepted_item_commas_separate_items(complement) is separates


@pytest.mark.parametrize(
    ("text", "enclosed", "phrase"),
    [
        (
            "subject to the lease.",
            False,
            "subject to the lease",
        ),
        (
            "subject to paragraph (b)) shall file.",
            True,
            "subject to paragraph (b)",
        ),
        ("subject to the request.\nand more", False, "subject to the request"),
        (
            "subject to rules (see page 5) here.",
            False,
            "subject to rules (see page 5) here",
        ),
        ("subject to rules (see 5 U.S.C. 552) here.", False, None),
        ("subject to MRRB. Fees apply.", False, None),
        ("subject to form 7. Fees apply.", False, None),
        ("subject to the request, if any.", False, None),
        ("subject to the request", False, None),
        ("subject to the request. And", True, None),
        ("subject to 12 U.S.C. 1841) shall file.", True, None),
        ("subject to the request) here.", False, None),
        ("subject to the request; or", False, None),
        ("subject to the request: a fee.", False, None),
        ("subject to the request -- or a fee.", False, None),
        ("subject to the\n(b) request.", False, None),
        ("subject to the U.S.", False, None),
        ("subject to the FDIC.", False, None),
    ],
)
def test_phrase_end(text, enclosed, phrase):
    end = sc._excepted_item_phrase_end(text, len("subject to"), enclosed=enclosed)
    assert end == (None if phrase is None else len(phrase))


@pytest.mark.parametrize(
    ("text", "breaks"),
    [
        ("\nThe fee", True),
        ("\n(b) the fee", True),
        ("\n§ 2 the fee", True),
        ("\n2 CFR", True),
        ("\n• wages", True),
        ("\n  (b) the fee", True),
        ("\na. each", True),
        ("\niv) each", True),
        ("\nabcde) each", True),
        ("\nabcdef) each", False),
        ("\n\neach", True),
        ("\n \neach", True),
        ("\neach grant", False),
        ("\napplicant's", False),
        ("\ntwo-year", False),
    ],
)
def test_line_break_starts_a_new_paragraph(text, breaks):
    assert (sc._EXCEPTED_ITEM_LINE_BREAK.match(text) is not None) is breaks


@pytest.mark.parametrize(
    ("lead", "found"),
    [
        ("The agency may not charge any fee ", True),
        ("The agency isn't charging any fee ", True),
        ("The agency may (in an emergency) charge any fee ", True),
        ("The agency charges any fee ", False),
        ("The agency (which may act) charges any fee ", False),
        ("The agency may charge fees (for any fee ", False),
        ("The agency may charge fees; any fee ", False),
        ("The agency may charge fees. Any fee ", False),
        ("The agency may charge fees\n(b) any fee ", False),
        ("The agency may charge fees under 12 U.S.C. 1709 any fee ", False),
        # A known limit: a verb in a relative clause counts.
        ("A lender that is a bank ", True),
    ],
)
def test_clause_before_other_than_needs_a_predicate(lead, found):
    text = lead + "other than a copy fee"
    assert sc._excepted_item_clause_has_predicate(text, len(lead)) is found


@pytest.mark.parametrize(
    ("word", "heads"),
    [
        ("address", True),
        ("deed", True),
        ("hundred", True),
        ("applicant's", True),
        ("one", True),
        ("another", True),
        ("listed", False),
        ("directly", False),
        ("only", False),
        ("need", False),
        ("itself", False),
        ("anyone", False),
        ("stays", False),
        ("The", False),
        # A noun ending in "-ly" stays on the safe side.
        ("family", False),
    ],
)
def test_word_before_the_cue_must_head_a_noun(word, heads):
    assert sc._excepted_item_can_head_noun(word) is heads


@pytest.mark.parametrize(
    ("before_cue", "word"),
    [
        ("a copy ", "copy"),
        ("a 'copy ", "copy"),
        ("a ’copy ", "copy"),
        ("a -copy ", "copy"),
        ("the two-year ", "two-year"),
        ("the applicant's ", "applicant's"),
        ("the applicant’s ", "applicant’s"),
        ("a copy\n", "copy"),
        ("a copy  ", "copy"),
        ("a 401k ", None),
        ("a 401-unit ", None),
        ("a 4'x ", None),
        ("a 4’x ", None),
        ("a copy, ", None),
        ("a copy” ", None),
        ("a copy", None),
    ],
)
def test_word_right_before_the_cue(before_cue, word):
    match = sc._EXCEPTED_ITEM_PREVIOUS_WORD.search(before_cue)
    assert (None if match is None else match.group(1)) == word


# Every member of each word list, spelled out here so that a member dropped
# from the module fails its own witness below. The condition terms are
# patterns, so their entry holds each spelling of each term.
WORD_LIST_MEMBERS = {
    "_EXCEPTED_ITEM_FRONTING_MARKS": [",", "-"],
    "_EXCEPTED_ITEM_NON_NOUNS": """
        a about above after against all also am among an and any anybody anyone
        anything appear appearing appears are as at be became become becomes
        becoming been before being below between beyond both but by can continue
        continues continuing could during each either every everybody everyone
        everything except fall falling falls for from grow growing grows having
        he held her hers herself him himself his if in including into is it its
        itself lie lies lying made may me mine must myself neither nobody none
        nor not nothing of on oneself onto or other otherwise our ours ourselves
        over per prove proves proving remain remaining remains rest resting
        rests seem seeming seems shall she should sit sits sitting somebody
        someone something stand standing stands stay staying stays such than
        that the their theirs them themself themselves these they this those
        through to toward turn turning turns under unless upon us via was we
        were whatever when where which whichever who whoever whom whomever whose
        will with within without would you your yours yourself yourselves
    """.split(),
    "_EXCEPTED_ITEM_ED_NOUNS": """
        bed breed creed deed feed greed hundred red reed seed shed speed steed
        weed
    """.split(),
    "_EXCEPTED_ITEM_PREDICATE_WORDS": """
        am are can cannot could dare did do does had has have is may might must
        need ought shall should was were will would
    """.split(),
    "_EXCEPTED_ITEM_CLAUSE_WORDS": """
        although because if since that though unless until when whenever where
        whereas whereby wherein wherever whether which while who whom whose
    """.split(),
    "_EXCEPTED_ITEM_PHRASE_LEADS": """
        after as at before beginning by during except for from if in including
        of on provided such that to under unless until upon when where whether
        which who whom whose with within without
    """.split(),
    "_EXCEPTED_ITEM_QUANTIFIERS": """
        all another any each either every neither no some such
    """.split(),
    "_EXCEPTED_ITEM_ABBREVIATIONS": """
        al amdt ann app approx art arts assn ave bros cf ch chap chs cir cl cmt
        co comm cong corp ct dept div doc dr ed eds est etc fed fig figs ft gov
        govt hr hrs ibid id inc jr lb lbs ltd max min mo mos mr mrs ms no nos op
        oz par para paras pp proc prop pt pts pub reg regs rep rept res rev rul
        sched sec secs seq sess sq sr st stat subch subd subpar subpt subsec
        supp temp tit tr treas viz vol vols vs yr yrs
    """.split(),
    "_EXCEPTED_ITEM_CONDITION_TERMS": """
        sec secs section sections subsection subsections paragraph paragraphs
        subparagraph subparagraphs clause clauses subclause subclauses item
        items chapter chapters subchapter subchapters title titles part parts
        subpart subparts article articles division divisions subdivision
        subdivisions schedule schedules appendix appendices exhibit exhibits
        annex heading headings subheading subheadings note notes this these
        those such said applicable any all each every provision provisions
        requirement requirements limitation limitations limit limits condition
        conditions term terms restriction restrictions rule rules regulation
        regulations law laws statute statutes act acts code codes order orders
        standard standards procedure procedures guideline guidelines policy
        policies criteria criterion qualification qualifications exception
        exceptions exemption exemptions waiver waivers maximum maximums minimum
        minimums cap caps ceiling ceilings threshold thresholds reduction
        reductions adjustment adjustments deduction deductions offset offsets
        withholding tax taxes taxation penalty penalties sanction sanctions
        forfeiture forfeitures recapture repayment repayments matching
        contribution contributions appropriation appropriations appropriated
        availability available fund funds funding budget budgets allotment
        allotments allocation allocations approval approvals consent concurrence
        authorization authorizations permission discretion direction
        determination determinations certification certifications verification
        verifications audit audits inspection inspections oversight supervision
        control jurisdiction authority enforcement agreement agreements notice
        notices hearing hearings appeal appeals secretary administrator director
        commissioner board commission court agency department governor president
        congress legislature comptroller bureau attorney officer official
        tribunal judge
    """.split(),
}
# Members no decision depends on, because another rule always decides as they
# do. A non-noun that the complement check's predicate or clause-word pattern
# lists is the last word of the complement whenever it comes right before the
# cue, and the complement check rejects it there; no letter, digit or
# underscore runs into it, so the whole-word patterns find it. A phrase lead
# that is also a clause word fails the complement anywhere, after a comma or
# not. And if the period of an abbreviation that is also a condition word
# ("sec.") ended the phrase, the phrase would end in that condition word and
# keep its cue all the same.
EQUIVALENT_MEMBERS = {
    "_EXCEPTED_ITEM_NON_NOUNS": frozenset(
        """
        am are can could if is may must shall should that unless was were when
        where which who whom whose will would
        """.split()
    ),
    "_EXCEPTED_ITEM_PHRASE_LEADS": frozenset(
        "if that unless until when where whether which who whom whose".split()
    ),
    "_EXCEPTED_ITEM_ABBREVIATIONS": frozenset({"sec", "secs"}),
}
# Every other member must change some decision, so each list has a sentence
# with "{}" where one member goes, and whether that sentence drops its cue
# while the list holds the member. The sentence must decide the other way once
# the member is gone.
WORD_LIST_WITNESSES = {
    "_EXCEPTED_ITEM_FRONTING_MARKS": (
        "Each registrar shall file every record{} other than the owner's deed "
        "and the parcel map subject to the lease.",
        False,
    ),
    "_EXCEPTED_ITEM_NON_NOUNS": (
        "The agency may not charge any fee other than a copy fee {} subject to review.",
        False,
    ),
    "_EXCEPTED_ITEM_ED_NOUNS": (
        "The clerk may not record any item other than a {} subject to review.",
        True,
    ),
    "_EXCEPTED_ITEM_PREDICATE_WORDS": (
        "The agency {lead} require that a lender other than a bank {} file "
        "reports subject to review.",
        False,
    ),
    "_EXCEPTED_ITEM_CLAUSE_WORDS": (
        "The agency may not charge any fee other than copies {} a person pays "
        "fees subject to review.",
        False,
    ),
    "_EXCEPTED_ITEM_PHRASE_LEADS": (
        "The agency may not charge any fee other than copies, {} a search fee, or "
        "a fee subject to review.",
        False,
    ),
    "_EXCEPTED_ITEM_QUANTIFIERS": (
        "The agency may not impose any fee other than a copy fee or {} charge "
        "subject to review.",
        False,
    ),
    "_EXCEPTED_ITEM_ABBREVIATIONS": (
        "The agency may not charge any fee other than a copy fee subject to "
        "review {}. Rules apply.",
        False,
    ),
    "_EXCEPTED_ITEM_CONDITION_TERMS": (
        "The Secretary shall pay benefits to persons other than veterans subject "
        "to {} review.",
        False,
    ),
}


def _witness(name: str, member: str) -> str:
    template, _drops = WORD_LIST_WITNESSES[name]
    word = " -" if member == "-" else member
    return template.format(word, lead="shall" if member == "may" else "may")


def _list_entry(name: str, member: str) -> str:
    """The entry of the module's list that covers ``member``."""

    if name != "_EXCEPTED_ITEM_CONDITION_TERMS":
        return member
    return next(term for term in getattr(sc, name) if re.fullmatch(term, member))


def _without_member(patch: pytest.MonkeyPatch, name: str, member: str) -> None:
    """Patch the module as if the list ``name`` lacked ``member``'s entry."""

    entry = _list_entry(name, member)
    words = getattr(sc, name)
    if isinstance(words, frozenset):
        patch.setattr(sc, name, words - {entry})
        return
    old = sc._excepted_item_words(words)
    new = sc._excepted_item_words(word for word in words if word != entry)
    patterns = [
        (attribute, value)
        for attribute, value in vars(sc).items()
        if attribute.startswith("_EXCEPTED_ITEM_")
        and isinstance(value, re.Pattern)
        and old in value.pattern
    ]
    assert patterns, name
    for attribute, value in patterns:
        patch.setattr(
            sc, attribute, re.compile(value.pattern.replace(old, new), value.flags)
        )


def _word_list_cases(*, equivalent: bool) -> list:
    return [
        pytest.param(name, member, id=f"{name}:{member}")
        for name, members in WORD_LIST_MEMBERS.items()
        for member in members
        if (member in EQUIVALENT_MEMBERS.get(name, ())) is equivalent
    ]


@pytest.mark.parametrize("name", WORD_LIST_MEMBERS)
def test_every_word_list_member_has_a_witness(name):
    words = getattr(sc, name)
    members = WORD_LIST_MEMBERS[name]
    assert len(members) == len(set(members))
    if name == "_EXCEPTED_ITEM_CONDITION_TERMS":
        # Each pattern matches some listed spelling, and each spelling exactly
        # one pattern, so a pattern that joins, leaves, or repeats spellings
        # another pattern covers ("sec" beside "secs?") fails here.
        assert len(words) == len(set(words))
        assert all(any(re.fullmatch(term, word) for word in members) for term in words)
        assert all(
            sum(re.fullmatch(term, word) is not None for term in words) == 1
            for word in members
        )
    else:
        assert len(words) == len(set(words))
        assert set(words) == set(members)


@pytest.mark.parametrize(("name", "member"), _word_list_cases(equivalent=False))
def test_each_word_list_member_decides_its_witness(name, member):
    text = _witness(name, member)
    drops = WORD_LIST_WITNESSES[name][1]
    assert list(_decisions(text).values()) == [drops]
    with pytest.MonkeyPatch.context() as patch:
        _without_member(patch, name, member)
        assert list(_decisions(text).values()) == [not drops]


def _helper_rejects(name: str, member: str) -> bool:
    """Whether the helper that reads ``name`` treats ``member`` as listed."""

    if name == "_EXCEPTED_ITEM_NON_NOUNS":
        return not sc._excepted_item_can_head_noun(member)
    if name == "_EXCEPTED_ITEM_PHRASE_LEADS":
        return not sc._excepted_item_commas_separate_items(
            f" copies, {member} a search fee, or a fee "
        )
    before = f"subject to review {member}"
    return not sc._excepted_item_sentence_ends_at(before + ". Rules", len(before))


ALL_TEXTS = [
    *SKIPPED.values(),
    *KEPT.values(),
    *KNOWN_LIMIT_DROPS.values(),
    *MINIMAL_TEXTS,
    *ENCLOSED_TEXTS,
    *(
        _witness(name, member)
        for name, members in WORD_LIST_MEMBERS.items()
        for member in members
    ),
]


@pytest.mark.parametrize(("name", "member"), _word_list_cases(equivalent=True))
def test_an_equivalent_member_changes_no_decision(name, member):
    assert _helper_rejects(name, member)
    before = [_decisions(text) for text in ALL_TEXTS]
    with pytest.MonkeyPatch.context() as patch:
        _without_member(patch, name, member)
        assert not _helper_rejects(name, member)
        assert [_decisions(text) for text in ALL_TEXTS] == before


def _exception_obligations(
    source: str, citation: str = CITATION
) -> tuple[sc.SourceStructureBranch, ...]:
    branches = tuple(
        sc.recognize_source_structure(source, corpus_citation_path=citation)
    )
    return sc._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        corpus_citation_path=citation,
    )


def test_the_paragraph_prohibition_has_no_exception_obligation():
    assert not any(
        "subject to the lease" in obligation.text
        for obligation in _exception_obligations(STALL_SOURCE)
    )
    control = [
        obligation
        for obligation in _exception_obligations(STALL_CONTROL)
        if "subject to the lease" in obligation.text
    ]
    assert len(control) == 1
    assert sc._source_exception_requires_paired_witness(
        control[0].text, branch_path=tuple(control[0].path)
    )


# The review's reproductions: printed text holds the "other than", so the
# phrase keeps its paired condition through the exception branches, as on
# main.
QUOTED_EXCEPTIONS = (
    "other than inside curly double quotes",
    "other than inside curly single quotes",
    "other than inside square brackets",
)


@pytest.mark.parametrize("name", QUOTED_EXCEPTIONS)
def test_a_quoted_other_than_keeps_the_paired_condition(name):
    text = KEPT[name]
    branches = tuple(sc.recognize_source_structure(text))
    obligations = sc._source_exception_branches(
        text,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        raw_source_text=text,
    )
    assert [
        sc._source_exception_condition_text(obligation.text)
        for obligation in obligations
        if sc._source_exception_requires_paired_witness(
            obligation.text, branch_path=tuple(obligation.path)
        )
    ] == ["subject to review."]


@pytest.mark.parametrize("source", CHAPEAUS.values(), ids=CHAPEAUS.keys())
def test_a_kept_cue_keeps_its_condition_text_in_its_clause(source):
    obligations = _exception_obligations(source, STATUTE_CITATION)
    assert len(obligations) == 1
    assert sc._source_exception_condition_text(obligations[0].text).startswith(
        "subject to the supervision of the Bureau"
    )


@pytest.mark.parametrize(
    "text",
    [
        *SKIPPED.values(),
        *KEPT.values(),
        *KNOWN_LIMIT_DROPS.values(),
        *MINIMAL_TEXTS,
        *ENCLOSED_TEXTS,
    ],
    ids=[
        *SKIPPED.keys(),
        *KEPT.keys(),
        *KNOWN_LIMIT_DROPS.keys(),
        *MINIMAL_PAIRS.keys(),
        *ENCLOSED_PAIRS.keys(),
    ],
)
def test_each_clause_keeps_a_cue_its_source_keeps(text):
    full = _decisions(text)
    for start, end, _clause in sc._source_clause_spans(text, branches=()):
        for offset, dropped in _decisions(text[start:end]).items():
            assert not dropped or full[start + offset], (start, end)


# Callers that collapse whitespace (_collapse_text) lose the line breaks the
# check reads. A capital in the complement, or a clause with no verb before
# the "other than", still keeps a heading's cue there.
def test_collapsed_heading_keeps_its_cue():
    raw = KEPT["heading without a period, short phrase"]
    assert list(_decisions(raw).values()) == [False]
    assert list(_decisions(sc._collapse_text(raw)).values()) == [False]


def test_collapsed_paragraph_label_can_drop_the_cue():
    # A known limit: once collapsed, a new paragraph whose label sits in
    # parentheses and whose text starts in lower case reads as part of the
    # list.
    raw = (
        "The agency may not charge for any service other than copies\n(b) each "
        "grant subject to review."
    )
    assert list(_decisions(raw).values()) == [False]
    assert list(_decisions(sc._collapse_text(raw)).values()) == [True]


def _analyze_stall_paragraph(source: str) -> list[str]:
    """Encode the paragraph with one single-input pair per listed input."""

    expression = " and ".join(
        name if index == 0 else f"not {name}" for index, name in enumerate(STALL_INPUTS)
    )
    rule = {
        "name": "stall_records_comply",
        "kind": "derived",
        "dtype": "Judgment",
        "source": CITATION + "(c)",
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": f"if {expression}: true else: false",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": source.strip(),
                        },
                    }
                ]
            }
        },
    }
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [{"name": name, "dtype": "Boolean"} for name in STALL_INPUTS],
        "rules": [rule],
    }
    compliant = {name: index == 0 for index, name in enumerate(STALL_INPUTS)}
    cases = [
        {
            "period": "2026",
            "input": compliant,
            "output": {"stall_records_comply": True},
        }
    ]
    for name in STALL_INPUTS:
        cases.append(
            {
                "period": "2026",
                "input": {**compliant, name: not compliant[name]},
                "output": {"stall_records_comply": False},
            }
        )
    analysis = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [str(issue) for issue in analysis.issues]


def test_faithful_encoding_of_the_paragraph_passes_the_complete_source_check():
    assert _analyze_stall_paragraph(STALL_SOURCE) == []


def test_same_qualifier_outside_an_other_than_list_still_needs_a_pair():
    issues = _analyze_stall_paragraph(STALL_CONTROL)
    assert any(
        "[complete-source-unit:tests]" in issue
        and "exceptions or applicability conditions" in issue
        and "subject to the lease" in issue
        for issue in issues
    ), issues


# Sentences of the shape the filter drops: a lead with a modal and a noun,
# "other than", a list of one to four items, the cue, a short object with no
# condition word, and a sentence end.
_LEADS = (
    "Each registrar shall file every record",
    "The agency may not charge for any service",
    "No fee may be imposed on any instrument",
    "External review must be conducted by a contractor",
)
# The same leads with no modal and no finite form of "be", "have" or "do". A
# lexical verb, as in "files" or "charges", doesn't count as a verb here.
_VERBLESS_LEADS = (
    "Each registrar files every record",
    "The agency charges for any service",
    "Each fee on any instrument",
    "External review by a contractor",
)
_ITEMS = (
    "the owner's deed",
    "the parcel map",
    "a copy",
    "a seasonal fee",
    "the record",
    "a deed",
    "the floor plan for the premises",
    "an annual fee",
)
_OBJECTS = (
    "the lease",
    "external review",
    "the request",
    "the owner's request",
    "review",
)
_ENDINGS = (".", ".\n", ". The board shall comply.", ".\n(f) Next.")


@st.composite
def _excepted_item_sentences(draw, connectives=("other than",), leads=_LEADS):
    items = draw(st.lists(st.sampled_from(_ITEMS), min_size=1, max_size=4))
    conjunction = draw(st.sampled_from(("and", "or", "nor")))
    if len(items) == 1:
        listed = items[0]
    elif len(items) == 2:
        listed = f"{items[0]} {conjunction} {items[1]}"
    else:
        listed = ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"
    return (
        f"{draw(st.sampled_from(leads))} {draw(st.sampled_from(connectives))} "
        f"{listed} subject to {draw(st.sampled_from(_OBJECTS))}"
        f"{draw(st.sampled_from(_ENDINGS))}"
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(text=_excepted_item_sentences())
def test_an_excepted_item_phrase_drops_exactly_its_cue(text):
    unfiltered = _unfiltered_cues(text)
    filtered = _cues(text)
    assert filtered <= unfiltered
    assert [cue for _start, _end, cue in unfiltered - filtered] == ["subject to"]


# Connectives a looser "other than" pattern could take for one.
_NEAR_MISSES = (
    "and",
    "including",
    "except",
    "rather than",
    "less than",
    "others than",
    "another than",
    "other-than",
    "otherthan",
    "other then",
    "Other than",
    "OTHER THAN",
)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(text=_excepted_item_sentences(connectives=_NEAR_MISSES))
def test_only_a_lowercase_other_than_lets_a_cue_drop(text):
    assert _cues(text) == _unfiltered_cues(text)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(text=_excepted_item_sentences(leads=_VERBLESS_LEADS))
def test_a_clause_with_no_verb_before_other_than_keeps_the_cue(text):
    assert _cues(text) == _unfiltered_cues(text)


_COMPLEMENT_BREAKERS = (
    "; ",
    ": ",
    " — ",
    " -- ",
    ". The ",
    ". (b) ",
    ". e ",
    " U.S. ",
    "\n(b) ",
    "\n• ",
    "\n\n",
    " Each ",
    " shall ",
    " is ",
    " cannot ",
    " isn't ",
    " ought ",
    " which ",
    " that ",
    " if ",
    " since ",
    " or any ",
    " and each ",
    " (",
    " ) ",
    ", after notice, ",
)
_PHRASE_BREAKERS = (
    ", ",
    "; ",
    ": ",
    " — ",
    " -- ",
    " shall ",
    " is ",
    " cannot ",
    " isn't ",
    "\n(b) ",
    "\na. ",
    " (",
    " ) ",
    " paragraph (b) ",
    " of the Secretary ",
    " under this section ",
    " by Acme ",
    " notice ",
    " hearing ",
)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(data=st.data())
def test_one_breaker_in_the_complement_or_phrase_keeps_the_cue(data):
    text = data.draw(_excepted_item_sentences())
    other_than = re.search(r"\bother\s+than\b", text)
    cue = re.search(r"\bsubject\s+to\b", text)
    assert other_than is not None and cue is not None
    if data.draw(st.booleans()):
        start, end, breakers = other_than.end(), cue.start(), _COMPLEMENT_BREAKERS
    else:
        start, end, breakers = cue.end(), text.index(".", cue.end()), _PHRASE_BREAKERS
    spaces = [index for index in range(start, end) if text[index] == " "]
    index = data.draw(st.sampled_from(spaces))
    changed = text[:index] + data.draw(st.sampled_from(breakers)) + text[index + 1 :]
    assert _kept_subject_to_spans(changed) == _subject_to_spans(changed)


_PREFIX_WORDS = (
    "The",
    "agency",
    "shall",
    "pay",
    "(a)",
    "fees",
    "vessels",
    "built",
    "in",
    "the",
    "copies",
    "other than",
    "(",
    ",",
    "U.S.",
    "subject to",
)
# Ways a prior sentence ends, among them an abbreviation whose period ends
# the sentence too.
_SENTENCE_ENDS = (". ", ".\n", "? ", "! ", ".” ", " U.S. ", " Inc. ", " et seq. ")
_CAPITALIZED_TEXTS = [
    text
    for text in [
        *SKIPPED.values(),
        *KEPT.values(),
        *KNOWN_LIMIT_DROPS.values(),
        *MINIMAL_TEXTS,
        *ENCLOSED_TEXTS,
    ]
    if re.match(r"[A-Z(]", text)
]


# Sentences with a cue and no "other than", so that only a prior sentence's
# "other than" could reach the cue.
_TEXTS_WITHOUT_OTHER_THAN = (
    "Owners file reports subject to review.",
    "Each grant subject to review accrues.",
    "Interest on a loan subject to review accrues daily.",
)
# An unfinished complement that a prior sentence can end with.
_OPEN_COMPLEMENTS = (
    "",
    " other than vessels built in the",
    " other than a loan made by",
    " may not charge any fee other than copies",
)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    text=st.sampled_from(_CAPITALIZED_TEXTS)
    | st.sampled_from(_TEXTS_WITHOUT_OTHER_THAN),
    prefix=st.lists(st.sampled_from(_PREFIX_WORDS), min_size=1, max_size=12).map(
        " ".join
    ),
    complement=st.sampled_from(_OPEN_COMPLEMENTS),
    end=st.sampled_from(_SENTENCE_ENDS),
)
def test_a_prior_sentence_leaves_the_decision_unchanged(text, prefix, complement, end):
    shifted = prefix + complement + end + text
    offset = len(shifted) - len(text)
    decisions = _decisions(text)
    shifted_decisions = {
        start - offset: dropped
        for start, dropped in _decisions(shifted).items()
        if start >= offset
    }
    assert shifted_decisions == decisions


def test_a_prior_sentence_that_leaves_a_quotation_open_keeps_the_cue():
    # The prior-sentence property holds for a sentence that leaves no
    # quotation or square bracket open. One that does puts the exception
    # inside the quotation, and the cue stays. So does a sentence with an
    # elided form whose apostrophe follows whitespace and comes before a
    # letter or digit, which the check reads as an opening single quote.
    assert list(_decisions(BASE).values()) == [True]
    for prior in (
        "He wrote “fees. ",
        "He wrote [fees. ",
        "He wrote ‘fees. ",
        "Rules from the '90s apply. ",
        "Rules from the ‘90s apply. ",
        "Rock 'n roll applies. ",
    ):
        assert list(_decisions(prior + BASE).values()) == [False], prior
    # A right single quote after whitespace closes nothing and opens nothing.
    assert list(_decisions("Rules from the ’90s apply. " + BASE).values()) == [True]


_ENDED_TEXTS = [text for text in _CAPITALIZED_TEXTS if text.endswith(".")]
# Sentences whose last period may end no sentence, so that a phrase which ran
# on past it would reach the following sentence.
_UNCERTAIN_PERIOD_TEXTS = tuple(
    BASE.replace("the lease.", ending)
    for ending in (
        "MRRB.",
        "the FDIC.",
        "form 7.",
        "the Dep't.",
        "the U.S.",
        "review etc.",
    )
)
_FOLLOWING_SENTENCES = (
    " The fee applies.",
    " Fees apply.",
    " (f) Next.",
    "\nThe fee applies.",
    "\n(f) Next.",
    "\n\nFees apply.",
    # Capitals outside ASCII, in Latin, Greek, Cyrillic and title case.
    " Élan acts.",
    " Łódź applies.",
    " Ωmega applies.",
    " Жилье applies.",
    " ǅemal acts.",
    "\nÉlan acts.",
)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    text=st.sampled_from(_ENDED_TEXTS) | st.sampled_from(_UNCERTAIN_PERIOD_TEXTS),
    following=st.sampled_from(_FOLLOWING_SENTENCES),
)
def test_a_following_sentence_leaves_the_decision_unchanged(text, following):
    decisions = _decisions(text)
    extended = _decisions(text + following)
    assert {start: extended[start] for start in decisions} == decisions


# A cut starts at a word and ends at a word boundary or at the end of the
# text. A cut that ends right after a mark sees a sentence end there that the
# full text may not, as "lease. 1. The" does, so the property leaves out
# the ends where the cut sees a sentence end the full text doesn't;
# test_a_cut_at_a_mark_the_text_runs_on_past_can_drop_more pins that limit.
_CUT_TAIL_MARK = re.compile(r"[.!?][\"'”’\]]*\s*\Z")
_CUT_TEXTS = [
    *SKIPPED.values(),
    *KEPT.values(),
    *KNOWN_LIMIT_DROPS.values(),
    *MINIMAL_TEXTS,
    *ENCLOSED_TEXTS,
]


def _cut_ends(text: str, start: int) -> list[int]:
    ends = []
    for end in [*(match.start() for match in re.finditer(r"\b", text)), len(text)]:
        if end <= start:
            continue
        cut = text[start:end]
        mark = _CUT_TAIL_MARK.search(cut)
        if (
            mark is not None
            and sc._excepted_item_sentence_ends_at(cut, mark.start())
            and not sc._excepted_item_sentence_ends_at(text, start + mark.start())
        ):
            continue
        ends.append(end)
    return ends


@settings(max_examples=300, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_word_bounded_cut_drops_no_cue_the_full_text_keeps(data):
    text = data.draw(st.sampled_from(_CUT_TEXTS) | _excepted_item_sentences())
    starts = [0, *(match.start() for match in re.finditer(r"\b(?=\w)", text))]
    other_than = re.search(r"\bother\s+than\b", text)
    cue = re.search(r"\bsubject\s+to\b", text, flags=re.IGNORECASE)
    around = (
        other_than is not None
        and cue is not None
        and other_than.start() < cue.start()
        and data.draw(st.booleans())
    )
    if around:
        # Keep the exception and the cue inside the cut, where a drop can
        # happen. A cut that starts at the "other than" can't drop the cue.
        starts = [start for start in starts if start < other_than.start()] or starts
    start = data.draw(st.sampled_from(starts))
    ends = _cut_ends(text, start)
    if around:
        # A cut that stops before the phrase's end can't drop the cue either.
        mark = re.compile(r"[.!?)]").search(text, cue.end())
        first = cue.end() if mark is None else mark.end()
        ends = [end for end in ends if end >= first] or ends
    end = data.draw(st.sampled_from(ends))
    full = _decisions(text)
    for offset, dropped in _decisions(text[start:end]).items():
        assert not dropped or full[start + offset], (start, end)


def test_a_cut_at_a_mark_the_text_runs_on_past_can_drop_more():
    text = (
        "Each registrar shall file every record other than the parcel map subject "
        "to the owner's two-year lease. 1. The registrar shall comply."
    )
    cue = text.index("subject to")
    assert _decisions(text) == {cue: False}
    assert _decisions(text[: text.index("1. The")]) == {cue: True}


# Quoted amendatory text that runs over more than one sentence. The full
# text, each clause the clause splitter cuts from it, and the exception
# branches all keep the cue.
QUOTED_SENTENCES = (
    "Section 5 is amended by inserting “The agency may not charge any fee other "
    "than a copy fee subject to review. Fees apply.”"
)


def test_a_quotation_across_sentences_keeps_the_cue_in_each_clause():
    cue = QUOTED_SENTENCES.index("subject to")
    assert _decisions(QUOTED_SENTENCES) == {cue: False}
    clauses = sc._source_clause_spans(QUOTED_SENTENCES, branches=())
    assert any(start <= cue < end for start, end, _clause in clauses)
    for start, end, _clause in clauses:
        for offset, dropped in _decisions(QUOTED_SENTENCES[start:end]).items():
            assert not dropped, (start, end, offset)
    branches = tuple(sc.recognize_source_structure(QUOTED_SENTENCES))
    obligations = sc._source_exception_branches(
        QUOTED_SENTENCES,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        raw_source_text=QUOTED_SENTENCES,
    )
    assert [
        sc._source_exception_condition_text(obligation.text)
        for obligation in obligations
        if sc._source_exception_requires_paired_witness(
            obligation.text, branch_path=tuple(obligation.path)
        )
    ] == ["subject to review."]


def test_a_cut_inside_a_quotation_can_drop_more():
    # A known limit of the cut property: a cut that starts after a quotation's
    # opening mark sees no open quotation, so it can drop a cue the full text
    # keeps, unless the quotation's closing mark falls inside the excepted
    # item, which runs to the phrase's sentence end and the closing marks
    # right after it, and isn't a single closing quote right after a letter
    # or digit, which the check reads as an apostrophe.
    cue = QUOTED_SENTENCES.index("subject to")
    start = QUOTED_SENTENCES.index("The agency")
    end = QUOTED_SENTENCES.index("Fees apply")
    assert _decisions(QUOTED_SENTENCES) == {cue: False}
    # The cut stops before the closing mark.
    assert _decisions(QUOTED_SENTENCES[start:end]) == {cue - start: True}
    # The cut keeps the closing mark, but after the next sentence.
    assert _decisions(QUOTED_SENTENCES[start:]) == {cue - start: True}
    # The closing mark falls inside the excepted item.
    one_sentence = QUOTED_SENTENCES.replace(" Fees apply.”", "”")
    assert one_sentence.endswith("subject to review.”")
    assert _decisions(one_sentence) == {cue: False}
    assert _decisions(one_sentence[start:]) == {cue - start: False}
    # A single closing quote after a letter falls inside the excepted item, in
    # the complement or in the phrase, but the cut reads it as an apostrophe.
    for closed in (
        "‘The agency may not charge any fee other than copies’ and records "
        "subject to review.",
        "‘The agency may not charge any fee other than copies subject to "
        "review’ and fees apply.",
    ):
        for single in (closed, closed.replace("‘", "'").replace("’", "'")):
            text = "Section 5 is amended by inserting " + single
            cue = text.index("subject to")
            start = text.index("The agency")
            assert _decisions(text) == {cue: False}, text
            assert _decisions(text[start:]) == {cue - start: True}, text


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    text=st.sampled_from(list(SKIPPED.values())),
    verb=st.sampled_from(
        [
            "shall",
            "may",
            "must",
            "will",
            "is",
            "are",
            "am",
            "has",
            "does",
            "cannot",
            "ought",
            "need",
            "isn't",
            "won’t",
        ]
    ),
)
def test_a_predicate_in_the_phrase_keeps_the_cue(text, verb):
    cue = next(re.finditer(r"\bsubject\s+to\b", text, flags=re.IGNORECASE))
    changed = text[: cue.end()] + f" what {verb} govern" + text[cue.end() :]
    assert _kept_subject_to_spans(changed) == _subject_to_spans(changed)


_OPENERS = ("“", "‘", "[", '"', "'", "«")
_CLOSERS = {"“": "”", "‘": "’", "[": "]", '"': '"', "'": "'", "«": "»"}


def _word_starts(text: str, start: int, end: int) -> list[int]:
    return [
        match.start()
        for match in re.finditer(r"\b(?=\w)", text)
        if start <= match.start() <= end
    ]


def _word_ends(text: str, start: int, end: int) -> list[int]:
    return [
        match.end() for match in re.finditer(r"\w+", text) if start < match.end() <= end
    ]


@settings(max_examples=300, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_excepted_item_across_a_quotation_edge_keeps_the_cue(data):
    # The invariant the quotation check restores: when the excepted item, from
    # the "other than" to its phrase's end, starts inside a quotation or square
    # bracket or crosses into or out of one, the cue stays. A lone single
    # closing quote after a letter reads as a possessive, so only the other
    # closing marks can cross out alone.
    text = data.draw(_excepted_item_sentences())
    other_than = re.search(r"\bother\s+than\b", text)
    cue = re.search(r"\bsubject\s+to\b", text)
    assert other_than is not None and cue is not None
    phrase_end = text.index(".", cue.end()) + 1
    shape = data.draw(st.sampled_from(("starts inside", "crosses in", "crosses out")))
    if shape == "crosses out":
        closer = data.draw(st.sampled_from(("”", "]", '"', "»")))
        at = data.draw(
            st.sampled_from(
                [*_word_ends(text, other_than.end(), cue.start() - 1), phrase_end]
            )
        )
        changed = text[:at] + closer + text[at:]
    else:
        opener = data.draw(st.sampled_from(_OPENERS))
        if shape == "starts inside":
            at = data.draw(st.sampled_from(_word_starts(text, 0, other_than.start())))
            closer_places = [
                None,
                *_word_ends(text, other_than.end(), cue.start() - 1),
                phrase_end,
            ]
        else:
            at = data.draw(
                st.sampled_from(_word_starts(text, other_than.end(), cue.start()))
            )
            closer_places = [None, phrase_end]
        close_at = data.draw(st.sampled_from(closer_places))
        changed = text
        if close_at is not None:
            changed = changed[:close_at] + _CLOSERS[opener] + changed[close_at:]
        changed = changed[:at] + opener + changed[at:]
    assert _kept_subject_to_spans(changed) == _subject_to_spans(changed), changed


@settings(max_examples=300, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_closed_quotation_or_possessive_in_the_complement_keeps_the_drop(data):
    # The check reads no quotation that opens and closes inside the
    # complement, and no possessive apostrophe, as an edge.
    text = data.draw(_excepted_item_sentences())
    other_than = re.search(r"\bother\s+than\b", text)
    cue = re.search(r"\bsubject\s+to\b", text)
    assert other_than is not None and cue is not None
    words = [
        match
        for match in re.finditer(r"[A-Za-z'-]+", text[: cue.start()])
        if match.start() > other_than.end()
        and match.group(0) not in {"and", "or", "nor"}
    ][:-1]
    word = data.draw(st.sampled_from(words))
    if data.draw(st.booleans()):
        opener = data.draw(st.sampled_from(_OPENERS))
        changed = (
            text[: word.start()]
            + opener
            + word.group(0)
            + _CLOSERS[opener]
            + text[word.end() :]
        )
    else:
        apostrophe = data.draw(st.sampled_from(("’", "'")))
        changed = text[: word.end()] + apostrophe + text[word.end() :]
    assert list(_decisions(changed).values()) == [True], changed
