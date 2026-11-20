"""Interactive command-line interface for the Educational Content Expander."""

from __future__ import annotations

import argparse
import sys

from .core import NGramModel, load_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="educational-expander",
        description="Expand a short educational phrase using a local n-gram model.",
    )
    parser.add_argument("--corpus", default="app/data", help=".txt file or directory of .txt files (default: app/data)")
    parser.add_argument("--order", type=int, default=3, help="n-gram order: 2 for bigram, 3 for trigram (default: 3)")
    parser.add_argument("--k", type=float, default=0.1, help="positive add-k smoothing value (default: 0.1)")
    parser.add_argument("--max-words", type=int, default=30, help="maximum generated continuation tokens (default: 30)")
    parser.add_argument("--seed", type=int, help="random seed for repeatable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = NGramModel(order=args.order, k=args.k, seed=args.seed).fit(load_corpus(args.corpus))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("Educational Content Expander")
    print(f"Loaded {model.vocabulary_size - 1} vocabulary words from {args.corpus}.")
    print("Enter a short phrase or bullet point; type 'quit' to exit.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if prompt.lower() in {"quit", "exit", ":q"}:
            print("Goodbye.")
            return 0
        if not prompt:
            print("Please enter a phrase to expand.")
            continue
        try:
            print(model.generate(prompt, max_words=args.max_words))
        except ValueError as exc:
            print(f"Could not generate: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
