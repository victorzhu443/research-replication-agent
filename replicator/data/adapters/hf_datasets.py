"""Hugging Face datasets adapter: probe by repo id via the Hub API; split sizes from the Hub's
dataset-info when available. Examples never reach a prompt. Downloads happen in reproduce.sh."""
from __future__ import annotations

import json
from urllib.parse import urlencode

import pandas as pd
from secprim.safe_fetch import safe_fetch

HUB = "https://huggingface.co/api/datasets/"
HOSTS = ["huggingface.co"]
MAX_API_BYTES = 5_000_000  # a dataset card with its info block is a few hundred KB at most

# canonical spec names -> Hub repo ids (grown by hand; the KB, not the model, owns this)
CANONICAL = {
    "wmt14_en_de": "wmt/wmt14", "wmt14_en_fr": "wmt/wmt14", "newstest2014": "wmt/wmt14", "newstest2013": "wmt/wmt14",
    "wmt16_en_de": "wmt/wmt16", "glue": "nyu-mll/glue", "sst2": "stanfordnlp/sst2", "squad": "rajpurkar/squad",
    "imagenet": "ILSVRC/imagenet-1k", "cifar10_hf": "uoft-cs/cifar10", "mnist_hf": "ylecun/mnist",
    "wikitext103": "Salesforce/wikitext", "penn_treebank": "ptb-text-only/ptb_text_only", "penn_treebank_wsj": "ptb-text-only/ptb_text_only",
    "librispeech": "openslr/librispeech_asr", "coco": "detection-datasets/coco",
    "glue_sst2": "stanfordnlp/sst2", "sst-2": "stanfordnlp/sst2", "glue_benchmark": "nyu-mll/glue",
    "imdb": "stanfordnlp/imdb", "anthropic_hh": "Anthropic/hh-rlhf", "hh_rlhf": "Anthropic/hh-rlhf",
    "cnn_dailymail": "abisee/cnn_dailymail", "samsum": "Samsung/samsum", "wikisql": "Salesforce/wikisql",
    "e2e_nlg": "tuetschek/e2e_nlg", "counterfact_dataset": "azhx/counterfact", "counterfact": "azhx/counterfact",
    "bookcorpus": "bookcorpus/bookcorpus", "tldr_reddit_summarization": "openai/summarize_from_feedback",
    "openwebtext": "Skylion007/openwebtext", "wikitext": "Salesforce/wikitext", "zsre": "azhx/zsre",
}

# Pretrained checkpoints on the Hub count as exact sources (checkpoint_eval family)
MODEL_HINTS = ("hf_pretrained", "huggingface_", "checkpoint", "pretrained_", "gpt2", "roberta", "bert", "llama", "t5")
MODEL_IDS = {"gpt2": "gpt2", "gpt2_medium": "gpt2-medium", "gpt2_xl": "gpt2-xl", "roberta_base": "roberta-base",
             "roberta_large": "roberta-large", "deberta_xxl": "microsoft/deberta-v2-xxlarge", "bert_base": "bert-base-uncased"}


def resolve_model(canonical: str) -> str | None:
    c = canonical.lower()
    if not any(h in c for h in MODEL_HINTS):
        return None
    for k, v in MODEL_IDS.items():
        if k in c:
            return v
    return "gpt2" if "gpt2" in c else None


def probe_model(repo: str) -> tuple[bool, str]:
    try:
        r = safe_fetch("https://huggingface.co/api/models/" + repo, allow_hosts=HOSTS, max_bytes=MAX_API_BYTES, timeout=20)
        return r.status_code == 200, f"HTTP {r.status_code} model {repo}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def resolve(canonical: str) -> str | None:
    if canonical in CANONICAL:
        return CANONICAL[canonical]
    if canonical.startswith("hf:"):
        return canonical[3:]
    return None


def probe(repo: str = "wmt/wmt14") -> tuple[bool, str]:
    try:
        r = safe_fetch(HUB + repo, allow_hosts=HOSTS, max_bytes=MAX_API_BYTES, timeout=20)
        if r.status_code == 200:
            j = _as_object(r.content)
            gated = j.get("gated", False)
            return (not gated), f"HTTP 200 {repo}" + (" (gated: needs HF token)" if gated else "")
        return False, f"HTTP {r.status_code} {repo}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _as_object(payload: bytes | str) -> dict:
    """Hub JSON must be an object; anything else is a declared ValueError."""
    j = json.loads(payload)
    if not isinstance(j, dict):
        raise ValueError("Hub response is not a JSON object")
    return j


def parse_dataset_info(payload: bytes | str) -> tuple[pd.DataFrame, str]:
    """Split sizes from a Hub dataset-info payload. Untrusted input: every shape is checked,
    and a malformed payload is a ValueError, never a crash."""
    j = _as_object(payload)
    card = j.get("cardData")
    infos = card.get("dataset_info") if isinstance(card, dict) else None
    if isinstance(infos, dict):
        infos = [infos]
    rows = []
    for cfg in infos if isinstance(infos, list) else []:
        if not isinstance(cfg, dict):
            continue
        splits = cfg.get("splits")
        for sp in splits if isinstance(splits, list) else []:
            if isinstance(sp, dict):
                rows.append({"config": cfg.get("config_name"), "split": sp.get("name"), "n_examples": sp.get("num_examples")})
    sha = j.get("sha")
    sha = sha[:12] if isinstance(sha, str) else ""
    return pd.DataFrame(rows or [{"config": None, "split": None, "n_examples": None}]), sha


def fetch(params: dict) -> tuple[pd.DataFrame, str]:
    repo = params["repo"]
    r = safe_fetch(HUB + repo + "?" + urlencode({"expand[]": "cardData"}), allow_hosts=HOSTS, max_bytes=MAX_API_BYTES, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} fetching dataset info for {repo}")
    df, sha = parse_dataset_info(r.content)
    return df, f"hf:{repo}:{sha}"
