import argparse
from pathlib import Path

from datasets import load_dataset

DATASETS = {
    "vifinqa": "AIGuruTinix/ViFinQA",
    "vietfintab": "VietFinTabGroup/VietFinTab",
    "vifintab-detection": "ngnquan/ViFinTab",
    "vnfinsqa": "duykhangh/VNFinsQA",
    "finqa": "dreamerdeo/finqa",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.dataset == "finqa":
        dataset = load_dataset(DATASETS[args.dataset], revision="refs/convert/parquet")
    else:
        dataset = load_dataset(DATASETS[args.dataset])
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in dataset.items():
        selected = rows.select(range(min(args.limit, len(rows)))) if args.limit else rows
        selected.to_json(args.output / f"{args.dataset}-{split}.jsonl", force_ascii=False)
        print(f"Saved {len(selected)} rows: {args.dataset}-{split}.jsonl")


if __name__ == "__main__":
    main()
