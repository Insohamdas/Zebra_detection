"""Bulk image download helpers for ZEBRAID.

This module uses :mod:`icrawler` to collect images for dataset bootstrapping.
Prefer reviewed, licensed, or public-domain sources before using the results
for training.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal


CrawlerEngine = Literal["google", "bing"]


def _load_crawler(engine: CrawlerEngine) -> Callable[..., Any]:
    """Return the icrawler class for the requested search engine."""

    try:
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    except ImportError as exc:  # pragma: no cover - exercised in real installs
        raise RuntimeError(
            "icrawler is not installed. Install it with 'pip install icrawler' "
            "or 'pip install -e .'."
        ) from exc

    crawlers: dict[CrawlerEngine, Callable[..., Any]] = {
        "google": GoogleImageCrawler,
        "bing": BingImageCrawler,
    }
    return crawlers[engine]


def download_images(
    keyword: str,
    output_dir: str | Path,
    *,
    max_num: int = 500,
    engine: CrawlerEngine = "google",
) -> Path:
    """Download images for a search phrase into ``output_dir``.

    Args:
        keyword: Search phrase used by the crawler.
        output_dir: Directory where images will be written.
        max_num: Maximum number of images to fetch.
        engine: Search backend to use. ``google`` matches the user example;
            ``bing`` is a useful fallback when Google rate-limits requests.

    Returns:
        The resolved output directory.
    """

    if max_num <= 0:
        raise ValueError("max_num must be a positive integer")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    crawler_cls = _load_crawler(engine)
    crawler = crawler_cls(storage={"root_dir": str(output_path)})
    crawler.crawl(keyword=keyword, max_num=max_num)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Bulk-download zebra images with icrawler"
    )
    parser.add_argument(
        "--keyword",
        default="zebra side view",
        help="Search phrase used for the crawl",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=500,
        help="Maximum number of images to download",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/zebra_images",
        help="Directory where downloaded images will be stored",
    )
    parser.add_argument(
        "--engine",
        choices=("google", "bing"),
        default="google",
        help="Crawler backend to use",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = download_images(
            args.keyword,
            args.output_dir,
            max_num=args.max_num,
            engine=args.engine,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    print(
        f"Downloaded up to {args.max_num} images for {args.keyword!r} "
        f"into {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())