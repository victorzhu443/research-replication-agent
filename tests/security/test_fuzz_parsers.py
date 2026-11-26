"""Fuzz the parsers that read untrusted paper text.

`find_companions` and `_arxiv_version` run over the text extracted from a paper
PDF — fully attacker-influenceable content (anyone can post a PDF). They must
degrade safely: return or raise a declared exception, never crash the run with
an unexpected error type. See ../../../security-agents/fuzzing/ for the method.
"""
from fuzzing.fuzz_lib import assert_robust

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
