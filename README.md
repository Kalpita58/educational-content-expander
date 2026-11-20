[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/td4m55spwm/badge.svg)](https://msrit.ask.thili.ai/b/td4m55spwm)

# Educational Content Expander

A small, local command-line tool that continues a short educational phrase or bullet point using a count-based n-gram language model. It reads UTF-8 text files from a corpus directory, preprocesses them, trains a trigram model by default, and samples a continuation with add-k smoothing.

## Setup

Requires Python 3.10 or newer. The generator itself uses only the Python standard library. To run the test suite, install the development dependency:

```sh
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

From the project directory:

```sh
python -m app.cli
```

Enter a phrase or bullet point at the prompt. Type `quit`, `exit`, or `:q` to leave. Use `Ctrl-C` or end-of-file to exit as well.

Options:

```text
python -m app.cli --corpus app/data --order 3 --k 0.1 --max-words 30 --seed 7
```

- `--corpus`: a `.txt` file or directory of `.txt` files (subdirectories are included); defaults to the included sample corpus in `app/data`.
- `--order`: n-gram order. `2` is a bigram model and `3` is a trigram model. Defaults to `3`.
- `--k`: positive add-k smoothing value. Defaults to `0.1`.
- `--max-words`: maximum continuation length (punctuation tokens count toward the limit). Defaults to `30`.
- `--seed`: optional random seed to make output repeatable.

You can also use it from Python:

```python
from app.core import NGramModel, load_corpus

model = NGramModel(order=3, k=0.1, seed=7).fit(load_corpus("app/data"))
print(model.generate("Photosynthesis uses sunlight", max_words=18))
```

To use your own subject material, place one or more plain text `.txt` files in a directory and pass it with `--corpus path/to/texts`. The included sample corpus covers biology, earth science, and study skills.

## How it works

Text is lowercased and tokenized into words and punctuation. Each sentence becomes a sequence of next-token counts, with sentence-start and sentence-end markers. At prediction time, the model estimates each candidate probability as `(context_count + k) / (all_context_counts + k × vocabulary_size)`. It uses the longest available context up to the selected n-gram order and backs off to shorter contexts when necessary. It then samples from that probability distribution. An end-of-sentence token may stop generation early.

## Milestones

- [x] Gather a subject-relevant educational corpus (`app/data/` includes biology, earth science, and study-skills samples).
- [x] Read UTF-8 `.txt` files from a single file or recursively from a directory; tokenize and normalize the text.
- [x] Build configurable n-gram counts and estimate next-token frequencies with add-k smoothing.
- [x] Generate continuations iteratively from short input, using context backoff when needed.
- [x] Provide an interactive CLI for phrase input and generated expansions.
- [x] Document setup, usage, and the limitations of generated text.

## Limitations

This is a simple statistical demonstration, not a grammar-aware writing system or a modern language model. It reuses patterns from the supplied corpus, so output quality depends on corpus size, quality, and subject coverage. Add-k smoothing prevents zero probabilities but also assigns probability to unseen tokens; it does not make the output factually correct. Sparse or mismatched prompts may yield awkward, repetitive, or irrelevant continuations. The model lowercases its learned tokens, does not preserve source formatting, and can produce fragments. Review every result for grammar, accuracy, and suitability before using it in lesson materials. Use only text you have permission to process.

## Tests

```sh
python -m pytest
```

The included tests check preprocessing, n-gram counts, add-k distributions, backoff, seeded generation, corpus loading, validation, and an interactive CLI smoke path.
