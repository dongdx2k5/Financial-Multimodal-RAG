"""HuggingFace Dataset Downloader for GraphRAG pipeline.

Downloads ViFinTab and FinQA datasets from HuggingFace Hub and caches
them as local Parquet files. Subsequent runs skip the download if cache
already exists.

Usage (CLI)
-----------
  python -m financial_rag.ingestion.hf_downloader
  python -m financial_rag.ingestion.hf_downloader --dataset finqa --split train
  python -m financial_rag.ingestion.hf_downloader --dataset vifintab --sample 500

Usage (Python)
--------------
  from financial_rag.ingestion.hf_downloader import HuggingFaceDatasetDownloader
  downloader = HuggingFaceDatasetDownloader(cache_dir=Path("data/raw/datasets/huggingface"))
  downloader.download_finqa()
  downloader.download_vifintab()
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from datasets import load_dataset   # type: ignore[import-untyped]
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------

FINQA_HF_ID    = "dreamerdeo/finqa"
VIFINTAB_HF_ID = "ngnquan/ViFinTab"

FINQA_SPLITS    = ("train", "validation", "test")
VIFINTAB_SPLITS = ("train", "validation")


# ---------------------------------------------------------------------------
# HuggingFaceDatasetDownloader
# ---------------------------------------------------------------------------

class HuggingFaceDatasetDownloader:
    """Download and cache HuggingFace datasets as local Parquet files.

    Parameters
    ----------
    cache_dir : Path
        Root cache directory. Parquet files are saved under
        ``<cache_dir>/finqa_parquet/`` and ``<cache_dir>/ViFinTab_parquet/``.
    force_redownload : bool
        If True, always re-download even if local cache exists.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        force_redownload: bool = False,
    ) -> None:
        from financial_rag.config.settings import get_settings
        settings = get_settings()
        self.cache_dir = Path(cache_dir or settings.hf_cache_dir)
        self.force = force_redownload

    # ------------------------------------------------------------------ #
    # FinQA
    # ------------------------------------------------------------------ #

    def download_finqa(
        self,
        splits: Sequence[str] = FINQA_SPLITS,
        sample_size: int | None = None,
    ) -> dict[str, Path]:
        """Download FinQA and save each split as a Parquet file.

        Parameters
        ----------
        splits : sequence of str
            Which splits to download: "train", "validation", "test".
        sample_size : int or None
            If set, only the first ``sample_size`` rows of each split
            are saved (useful for development).

        Returns
        -------
        dict mapping split name → local Parquet path.
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "The 'datasets' package is required. Install it with:\n"
                "  pip install datasets"
            )

        out_dir = self.cache_dir / "finqa_parquet"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for split in splits:
            out_path = out_dir / f"{split}.parquet"
            if out_path.exists() and not self.force:
                print(f"[FinQA] Cache hit: {out_path}")
                paths[split] = out_path
                continue

            print(f"[FinQA] Downloading split='{split}' from {FINQA_HF_ID} …")
            ds = load_dataset(FINQA_HF_ID, split=split)
            if sample_size:
                ds = ds.select(range(min(sample_size, len(ds))))

            df = ds.to_pandas()
            df.to_parquet(out_path, index=False)
            print(f"[FinQA] Saved {len(df):,} rows → {out_path}")
            paths[split] = out_path

        return paths

    # ------------------------------------------------------------------ #
    # ViFinTab
    # ------------------------------------------------------------------ #

    def download_vifintab(
        self,
        splits: Sequence[str] = VIFINTAB_SPLITS,
        sample_size: int | None = None,
        include_images: bool = False,
    ) -> dict[str, Path]:
        """Download ViFinTab and save each split as a Parquet file.

        Parameters
        ----------
        splits : sequence of str
            Which splits to download: "train", "validation".
        sample_size : int or None
            Limit rows per split.
        include_images : bool
            If False (default), the ``image_base64`` column is dropped to
            save disk space — it is very large (often > 1 MB per row).
            Set True if you need the raw images.

        Returns
        -------
        dict mapping split name → local Parquet path.
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "The 'datasets' package is required. Install it with:\n"
                "  pip install datasets"
            )

        out_dir = self.cache_dir / "ViFinTab_parquet"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for split in splits:
            out_path = out_dir / f"{split}.parquet"
            if out_path.exists() and not self.force:
                print(f"[ViFinTab] Cache hit: {out_path}")
                paths[split] = out_path
                continue

            print(f"[ViFinTab] Downloading split='{split}' from {VIFINTAB_HF_ID} …")
            ds = load_dataset(VIFINTAB_HF_ID, split=split)
            if sample_size:
                ds = ds.select(range(min(sample_size, len(ds))))

            df = ds.to_pandas()

            # Drop image_base64 unless explicitly requested (saves ~95% disk)
            if not include_images and "image_base64" in df.columns:
                df = df.drop(columns=["image_base64"])
                print("[ViFinTab] Dropped 'image_base64' column to save disk space.")

            df.to_parquet(out_path, index=False)
            print(f"[ViFinTab] Saved {len(df):,} rows → {out_path}")
            paths[split] = out_path

        return paths

    # ------------------------------------------------------------------ #
    # Convenience: download both datasets at once
    # ------------------------------------------------------------------ #

    def download_all(
        self,
        finqa_sample: int | None = None,
        vifintab_sample: int | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Download FinQA + ViFinTab and return all paths."""
        return {
            "finqa":    self.download_finqa(sample_size=finqa_sample),
            "vifintab": self.download_vifintab(sample_size=vifintab_sample),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ViFinTab and/or FinQA from HuggingFace Hub."
    )
    parser.add_argument(
        "--dataset", choices=["all", "finqa", "vifintab"], default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--split", default=None,
        help="Specific split to download (e.g. train). Default: all splits.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit number of rows per split (useful for testing).",
    )
    parser.add_argument(
        "--include-images", action="store_true",
        help="Include base64-encoded images in ViFinTab (very large!).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if local cache exists.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Override cache directory path.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    downloader = HuggingFaceDatasetDownloader(
        cache_dir=cache_dir,
        force_redownload=args.force,
    )

    splits_finqa    = (args.split,) if args.split else FINQA_SPLITS
    splits_vifintab = (args.split,) if args.split else VIFINTAB_SPLITS

    if args.dataset in ("all", "finqa"):
        downloader.download_finqa(splits=splits_finqa, sample_size=args.sample)

    if args.dataset in ("all", "vifintab"):
        downloader.download_vifintab(
            splits=splits_vifintab,
            sample_size=args.sample,
            include_images=args.include_images,
        )

    print("\n✅ Download complete.")


if __name__ == "__main__":
    _main()
