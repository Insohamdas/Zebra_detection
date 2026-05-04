"""Build next curated YOLO dataset version from a baseline + mined hard examples."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build next dataset version with hard examples")
    parser.add_argument("--base-dataset", required=True, help="Path to baseline YOLO dataset root")
    parser.add_argument("--hard-mining-root", required=True, help="Path containing hard-mining groups")
    parser.add_argument("--output-dataset", required=True, help="Path for new dataset version")
    parser.add_argument("--prefix", default="hm", help="Filename prefix for added hard examples")
    parser.add_argument(
        "--groups",
        default="false_positive,false_negative,low_conf_true_positive",
        help="Comma-separated hard-mining groups to include",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output dataset if it exists")
    return parser.parse_args()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_base_dataset(base_dataset: Path, output_dataset: Path, force: bool) -> None:
    if output_dataset.exists():
        if not force:
            raise FileExistsError(f"Output dataset already exists: {output_dataset}")
        shutil.rmtree(output_dataset)
    shutil.copytree(base_dataset, output_dataset)


def normalize_label_to_single_class(label_path: Path) -> str:
    lines_out: list[str] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        parts[0] = "0"
        lines_out.append(" ".join(parts[:5]))
    return ("\n".join(lines_out) + "\n") if lines_out else ""


def existing_hashes(image_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for image_path in sorted(p for p in image_dir.iterdir() if p.is_file()):
        hashes.add(md5(image_path))
    return hashes


def write_data_yaml(dataset_root: Path) -> None:
    text = (
        f"path: {dataset_root}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['zebra']\n"
    )
    (dataset_root / "data.yaml").write_text(text)


def main() -> None:
    args = parse_args()
    base_dataset = Path(args.base_dataset)
    hard_mining_root = Path(args.hard_mining_root)
    output_dataset = Path(args.output_dataset)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    clone_base_dataset(base_dataset, output_dataset, args.force)

    train_images = output_dataset / "train" / "images"
    train_labels = output_dataset / "train" / "labels"
    seen_hashes = existing_hashes(train_images)

    added = 0
    skipped_dup = 0
    missing_label = 0
    group_counts: dict[str, int] = {g: 0 for g in groups}

    for group in groups:
        group_img_dir = hard_mining_root / group / "images"
        group_lbl_dir = hard_mining_root / group / "labels"
        if not group_img_dir.exists():
            continue

        for image_path in sorted(p for p in group_img_dir.iterdir() if p.is_file()):
            label_path = group_lbl_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_label += 1
                continue

            digest = md5(image_path)
            if digest in seen_hashes:
                skipped_dup += 1
                continue

            target_stem = f"{args.prefix}_{group}_{image_path.stem}"
            target_image = train_images / f"{target_stem}{image_path.suffix.lower()}"
            target_label = train_labels / f"{target_stem}.txt"

            shutil.copy2(image_path, target_image)
            target_label.write_text(normalize_label_to_single_class(label_path))

            seen_hashes.add(digest)
            added += 1
            group_counts[group] += 1

    write_data_yaml(output_dataset)

    summary = [
        f"base_dataset={base_dataset}",
        f"hard_mining_root={hard_mining_root}",
        f"output_dataset={output_dataset}",
        f"added_train_samples={added}",
        f"skipped_duplicates={skipped_dup}",
        f"missing_labels={missing_label}",
    ]
    for group, count in group_counts.items():
        summary.append(f"added_{group}={count}")

    summary_path = output_dataset / "merge_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")
    print(summary_path.read_text().strip())


if __name__ == "__main__":
    main()
