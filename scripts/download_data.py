"""Download Quick, Draw! numpy bitmaps and the MediaPipe hand landmarker model."""

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import quote

import requests

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from air_drawing import config  # noqa: E402

logger = logging.getLogger("download_data")


def category_url(category: str) -> str:
    """Return the Quick, Draw! numpy bitmap URL for a category."""
    return f"{config.QUICKDRAW_BASE_URL}{quote(category)}.npy"


def category_path(category: str) -> Path:
    """Return the local path of a category's .npy file."""
    return config.RAW_DATA_DIR / f"{category}.npy"


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, skipping files that already exist."""
    if dest.exists():
        logger.info("Skipping %s (already exists)", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=config.DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        done = 0
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=config.DOWNLOAD_CHUNK_BYTES):
                fh.write(chunk)
                done += len(chunk)
    tmp.replace(dest)
    logger.info("  saved %.1f MB", done / config.BYTES_PER_MB)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(config.CATEGORIES),
        choices=config.CATEGORIES,
        metavar="CATEGORY",
        help="Categories to download (default: all).",
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="Do not download hand_landmarker.task."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download everything requested; return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    jobs = [(category_url(c), category_path(c)) for c in args.categories]
    if not args.skip_model:
        jobs.append((config.HAND_LANDMARKER_URL, config.HAND_LANDMARKER_PATH))
    failures = 0
    for url, dest in jobs:
        try:
            download(url, dest)
        except requests.RequestException as exc:
            failures += 1
            logger.error("Failed to download %s: %s", url, exc)
    if failures:
        logger.error("%d download(s) failed", failures)
        return 1
    logger.info("All downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
