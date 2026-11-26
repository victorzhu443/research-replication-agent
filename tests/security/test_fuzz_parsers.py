"""Fuzz the parsers that read untrusted paper text.

`find_companions` and `_arxiv_version` run over the text extracted from a paper
PDF — fully attacker-influenceable content (anyone can post a PDF). They must
degrade safely: return or raise a declared exception, never crash the run with
an unexpected error type. See ../../../security-agents/fuzzing/ for the method.
"""
import json

import pandas as pd
from fuzzing.fuzz_lib import assert_robust, size_sweep
from hypothesis import strategies as st
from secprim.bounded import redos_probe

from replicator.data.adapters import french, hf_datasets
from replicator.spec import intake


def test_find_companions_safe_on_hostile_text():
    # Pure regex over paper text; a bad input must not raise anything unexpected.
    assert_robust(intake.find_companions, allowed=(ValueError, TypeError))()


def test_arxiv_version_safe_on_hostile_text():
    def parse(s: str):
        return intake._arxiv_version(s, known_id=None)

    parse.__name__ = "_arxiv_version"
    assert_robust(parse, allowed=(ValueError, TypeError))()


def test_arxiv_version_safe_with_attacker_controlled_known_id():
    # known_id is interpolated into a regex via re.escape; fuzz it too, since a
    # future caller could pass a less-sanitized id.
    def parse(s: str):
        return intake._arxiv_version("arXiv:2401.00001v3 [cs.LG] 1 Jan 2024", known_id=s)

    parse.__name__ = "_arxiv_version_known_id"
    assert_robust(parse, allowed=(ValueError, TypeError), max_examples=200)()


# ---- parsers the first pass missed (RC-2): the HTML strip stage and the two data adapters

def test_html_to_text_safe_on_hostile_markup():
    assert_robust(intake.html_to_text, allowed=(ValueError, TypeError))()


def test_html_to_text_patterns_are_linear():
    """The old `<(script|style)[^>]*>.*?</\\1>` and `<[^>]+>` were quadratic on hostile HTML."""
    for rx in (intake._TAG, intake._SS_OPEN, intake._NBSP, intake._HSPACE, intake._BLANKS):
        redos_probe(rx, n=20_000, deadline_s=0.1, probes=[("lt", "<"), ("lt-script", "<script"), ("gt", ">"),
                                                        ("ws", " \t\n"), ("nl", "\n"), ("amp", "&nbsp"), ("mixed", "<a> \n")])


def test_html_to_text_cost_grows_linearly():
    rows = size_sweep(intake.html_to_text, sizes=(10_000, 100_000, 1_000_000),
                      units=("<", "<script", "<script></script>", "<a href=x>", " \n", "&nbsp;", "a"), deadline_s=1.0)
    assert rows


def test_strip_script_style_semantics():
    assert intake.html_to_text("a<script>x<b>y</b></script>b<STYLE>z</style>c") == "a b c"
    assert intake.html_to_text("keep<script>dropped to the end") == "keep"


def test_french_parse_block_safe_on_hostile_text():
    def parse(s: str):
        return french._parse_block(s, 0)

    parse.__name__ = "french._parse_block"
    assert_robust(parse, allowed=(ValueError, TypeError, KeyError, IndexError, pd.errors.ParserError), max_examples=200)()


_json_like = st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20),
                          lambda c: st.lists(c, max_size=5) | st.dictionaries(st.text(max_size=12), c, max_size=5),
                          max_leaves=25)
_hub_payload = st.one_of(
    st.text(),                                  # not even JSON
    _json_like.map(json.dumps),                 # any JSON shape
    st.fixed_dictionaries({"sha": _json_like, "cardData": st.fixed_dictionaries({"dataset_info": _json_like})}).map(json.dumps),
)


def test_hf_dataset_info_parser_safe_on_hostile_payloads():
    """`fetch`'s response handling, fed fake Hub payloads (no network)."""
    assert_robust(hf_datasets.parse_dataset_info, allowed=(ValueError, TypeError), strategy=_hub_payload)()
