from pathlib import Path

import pytest

from app.core import NGramModel, load_corpus, preprocess


def test_preprocess_splits_sentences_and_normalizes():
    assert preprocess("Cells are alive. PLANTS grow!") == [
        ["cells", "are", "alive", "."], ["plants", "grow", "!"]
    ]


def test_fit_counts_ngram_contexts_and_add_k_probabilities():
    model = NGramModel(order=2, k=0.5, seed=1).fit(["cells grow. cells grow. cells divide."])
    assert model.counts[("cells",)]["grow"] == 2
    probs = model.probabilities(["cells"])
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["grow"] > probs["divide"]
    assert all(value > 0 for value in probs.values())


def test_unseen_context_backs_off_and_generation_is_seeded():
    texts = ["plants use sunlight. plants make food."]
    first = NGramModel(order=3, seed=12).fit(texts)
    second = NGramModel(order=3, seed=12).fit(texts)
    assert sum(first.probabilities(["unseen", "context"]).values()) == pytest.approx(1.0)
    assert first.generate("Plants use", max_words=8) == second.generate("Plants use", max_words=8)


def test_load_corpus_reads_txt_directory(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Cells grow.", encoding="utf-8")
    (tmp_path / "b.md").write_text("ignored", encoding="utf-8")
    assert load_corpus(tmp_path) == ["Cells grow."]


def test_invalid_parameters_and_empty_text():
    with pytest.raises(ValueError):
        NGramModel(order=0)
    with pytest.raises(ValueError):
        NGramModel(k=0)
    with pytest.raises(ValueError):
        NGramModel().fit(["   "])
