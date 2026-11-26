"""RC-8 reader/actor split: the model that holds build/run tools (the Builder) is fed the
schema-validated `Spec`, never the paper as a path or raw text. The paper is read by the tool-less
extractor; only its validated output reaches the actor."""
from __future__ import annotations

import inspect
import typing

from replicator.build import builder as builder_mod
from replicator.build.builder import Builder
from replicator.schema import Spec


def test_builder_entry_point_takes_spec_not_paper():
    hints = typing.get_type_hints(Builder.__init__)
    assert hints["spec"] is Spec
    params = inspect.signature(Builder.__init__).parameters
    assert "spec" in params
    for name, hint in hints.items():
        assert hint is not str, f"Builder.__init__({name}: str) would be a raw-text channel into the actor"
    assert not {"pdf", "paper", "paper_text", "pages", "text"} & set(params)
    assert list(inspect.signature(Builder.run).parameters) == ["self"]


def test_builder_prompts_are_built_from_the_spec():
    src = inspect.getsource(builder_mod)
    assert "paper_block" not in src and "pymupdf" not in src and "fetch_" not in src
    assert "model_dump_json" in inspect.getsource(Builder.run)
