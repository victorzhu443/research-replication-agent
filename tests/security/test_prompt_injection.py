"""Prompt-injection robustness for the paper-text input path.

Deterministic and cost-free: no model is called. We model the app's sink — paper
text concatenated into a summarize prompt — and prove two things:

1. Undefended, the documented injection corpus subverts it (the risk is real).
2. Routed through `replicator.security.harden_paper_text`, the corpus is resisted.

This is the measure→mitigate→verify loop from security-agents/promptinject,
bound to THIS agent's untrusted channel.
"""
from promptinject.corpus import CANARY, corpus
from promptinject.harness import evaluate

from replicator.security import harden_paper_text, scrub_model_output


def _sink(paper_text: str) -> str:
    """Stand-in for the paper-text prompt sink. A vulnerable app effectively lets
    embedded instructions through; we emulate that by surfacing the text, so a
    canary in the paper reaches 'output'."""
    return f"Summarize this paper for replication:\n{paper_text}"


def _defended(paper_text: str) -> str:
    return scrub_model_output(_sink(harden_paper_text(paper_text)))


def test_corpus_lands_on_undefended_sink():
    # Sanity: the threat is real; most cases subvert the naive sink.
    report = evaluate(_sink, corpus())
    assert report.landed, "corpus should expose the undefended sink"


def test_hardened_paper_text_resists_corpus():
    report = evaluate(_defended, corpus())
    assert report.ok, "\n" + report.render()


def test_control_document_is_not_flagged():
    # The benign control must pass through hardening without a canary.
    assert CANARY not in _defended("Ordinary paragraph about batch normalization.")


def test_paper_reading_prompts_state_the_data_boundary():
    # The real defense for this app is a system-prompt directive telling the model
    # that the paper (passed as a document block) is data, not instructions. The
    # three paper-reading prompts (extract, referee, page re-read) must carry it, or a hostile paper could steer the
    # extraction/verdict.
    from replicator.spec.extract import EXTRACT_SYSTEM, REFEREE_SYSTEM, REREAD_SYSTEM

    for prompt in (EXTRACT_SYSTEM, REFEREE_SYSTEM, REREAD_SYSTEM):
        low = prompt.lower()
        assert "untrusted" in low or "not a source of instructions" in low \
            or "never a source of instructions" in low, \
            "paper-reading system prompt is missing the data-boundary directive"
        assert "instruction" in low
