"""Deprecated entry point: the answering service no longer calls a language model."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", default="work/ollama-smoke.json")
    parser.add_argument("--question", help="Run one custom narrative question")
    parser.parse_args()
    parser.error(
        "Retired: /ask is deterministic and never calls Ollama. "
        "Use pytest and scripts/evaluate_vnm.py for the supported VNM report."
    )


if __name__ == "__main__":
    main()
