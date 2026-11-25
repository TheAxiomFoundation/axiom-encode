"""Grounding numbers that Chinese statutes write in characters.

Taiwan's Statute for Industrial Innovation (產業創新條例) Article 10-1 states
its spend band as 新臺幣一百萬元以上、二十億元以下, its rates as 百分之五,
百分之三 and 百分之三十, and its window from 中華民國一百十四年一月一日. No ASCII
digit appears, so before this reader every rate and threshold an encoding of
the article states was an ungrounded literal.
"""

import pytest

from axiom_encode.harness.validator_pipeline import (
    _cjk_numeral_value,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
)

ART_10_1_CITATION = "tw/statute/j0040051/10-1"
# Excerpts of the corpus body of ART_10_1_CITATION (law.moj.gov.tw, pcode
# J0040051), verbatim.
ART_10_1_TEXT = (
    "最近三年內無違反環境保護、勞工或食品安全衛生相關法律且情節重大情事之公司或有限合夥事業，"
    "自中華民國一百十四年一月一日起至一百十八年十二月三十一日止投資於自行使用之全新智慧機械、"
    "人工智慧產品或服務、節能減碳之相關全新硬體、軟體、技術或技術服務，其支出金額在同一課稅年度內"
    "合計達新臺幣一百萬元以上、二十億元以下之範圍，得選擇以下列方式之一抵減應納營利事業所得稅額，"
    "一經擇定不得變更。其各年度投資抵減金額以不超過當年度應納營利事業所得稅額百分之三十為限：\n"
    "一、於支出金額百分之五限度內，抵減當年度應納營利事業所得稅額。\n"
    "二、於支出金額百分之三限度內，自當年度起三年內抵減各年度應納營利事業所得稅額。\n"
    "公司或有限合夥事業於同一年度合併適用前項投資抵減及其他投資抵減時，其當年度合計得抵減總額"
    "以不超過當年度應納營利事業所得稅額百分之五十為限。"
)


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("五", 5),
        ("十", 10),
        ("十八", 18),
        ("三十", 30),
        ("一百十四", 114),
        ("一百十八", 118),
        ("一百萬", 1_000_000),
        ("十億", 1_000_000_000),
        ("二十億", 2_000_000_000),
        ("一億二千萬", 120_000_000),
        ("兩千", 2000),
        ("一千零五", 1005),
    ],
)
def test_character_numerals_read_as_their_values(raw, value):
    assert _cjk_numeral_value(raw) == value


def test_article_10_1_grounds_its_rates_band_and_window():
    values = extract_numbers_from_text(ART_10_1_TEXT)
    for expected in (0.05, 0.03, 0.30, 0.50, 1_000_000, 2_000_000_000, 3):
        assert any(abs(value - expected) < 1e-9 for value in values), expected
    # 民國一百十四年 and 一百十八年 are the Gregorian years 2025 and 2029.
    assert 2025 in values
    assert 2029 in values


def test_only_stated_rates_and_amounts_enter_the_inventory():
    inventory = extract_numeric_occurrences_from_text(ART_10_1_TEXT)
    assert sorted(inventory) == sorted(
        [0.30, 0.05, 0.03, 0.50, 1_000_000, 2_000_000_000]
    )


ART_10_1_MODULE = """format: rulespec/v1
rules:
- name: minimum_qualifying_investment_spend
  kind: parameter
  dtype: Money
  unit: TWD
  versions:
  - effective_from: '2025-01-01'
    formula: '1000000'
- name: maximum_qualifying_investment_spend
  kind: parameter
  dtype: Money
  unit: TWD
  versions:
  - effective_from: '2025-01-01'
    formula: '2000000000'
- name: current_year_credit_rate
  kind: parameter
  dtype: Rate
  versions:
  - effective_from: '2025-01-01'
    formula: '0.05'
- name: three_year_credit_pool_rate
  kind: parameter
  dtype: Rate
  versions:
  - effective_from: '2025-01-01'
    formula: '0.03'
- name: annual_credit_tax_cap_rate
  kind: parameter
  dtype: Rate
  versions:
  - effective_from: '2025-01-01'
    formula: '0.30'
- name: combined_investment_credit_tax_cap_rate
  kind: parameter
  dtype: Rate
  versions:
  - effective_from: '2025-01-01'
    formula: '0.50'
"""


def test_an_encoding_of_article_10_1_has_no_ungrounded_literals():
    assert (
        find_ungrounded_numeric_issues(
            ART_10_1_MODULE,
            source_text=ART_10_1_TEXT,
            source_citation_path=ART_10_1_CITATION,
        )
        == []
    )


def test_a_rate_the_article_does_not_state_stays_ungrounded():
    module = ART_10_1_MODULE.replace("formula: '0.05'", "formula: '0.07'")
    issues = find_ungrounded_numeric_issues(
        module,
        source_text=ART_10_1_TEXT,
        source_citation_path=ART_10_1_CITATION,
    )
    assert issues
