"""Build and audit the immutable RIGA+ labeled-image manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


DOMAINS = ("BinRushed", "Magrabia", "MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3")
SEED = 20260904


def domain_of(path: Path) -> str:
    for part in path.parts:
        if part in DOMAINS:
            return part
    raise ValueError(f"No domain in {path}")


def stable_fraction(sample_id: str, fold: str) -> float:
    key = f"{SEED}:{fold}:{sample_id}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16) / 2**64


def main(dataset_root: str, output_dir: str) -> None:
    root = Path(dataset_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    image_root, mask_root = root / "RIGA", root / "RIGA-mask"
    images = [p for p in image_root.rglob("*.tif") if "Unlabeled" not in p.parts]
    rows, errors = [], []
    for image_path in sorted(images):
        domain = domain_of(image_path)
        stem = image_path.stem
        candidate_dir = mask_root.joinpath(*image_path.relative_to(image_root).parts[:-1])
        masks = sorted(candidate_dir.glob(f"{stem}-*.tif"), key=lambda p: int(re.search(r"-(\d+)$", p.stem).group(1)))
        annotators = [int(re.search(r"-(\d+)$", p.stem).group(1)) for p in masks]
        with Image.open(image_path) as im:
            width, height = im.size
        mask_shapes = []
        for p in masks:
            with Image.open(p) as m:
                mask_shapes.append(m.size)
        if annotators != [1, 2, 3, 4, 5, 6]:
            errors.append(f"annotators:{image_path}:{annotators}")
        if any(shape != (width, height) for shape in mask_shapes):
            errors.append(f"shape:{image_path}:{mask_shapes}:{width}x{height}")
        sample_id = f"{domain}/{stem}"
        rows.append({
            "sample_id": sample_id,
            "domain": domain,
            "image_relpath": str(image_path.relative_to(root)),
            "mask_relpaths": ";".join(str(p.relative_to(root)) for p in masks),
            "n_annotators": len(masks),
            "width": width,
            "height": height,
            "group_id": sample_id,
        })

    fieldnames = list(rows[0])
    with (out / "riga_labeled_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    fold_counts = defaultdict(Counter)
    for target in DOMAINS:
        fold_rows = []
        for row in rows:
            split = "test" if row["domain"] == target else ("val" if stable_fraction(row["sample_id"], target) < 0.15 else "train")
            fold_counts[target][split] += 1
            fold_rows.append({**row, "outer_target": target, "split": split})
        with (out / f"riga_lodo_{target}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
            w.writeheader(); w.writerows(fold_rows)

    counts = Counter(r["domain"] for r in rows)
    audit = {
        "status": "PASS" if not errors and len(rows) == 744 and all(r["n_annotators"] == 6 for r in rows) else "FAIL",
        "dataset_root": str(root),
        "labeled_images": len(rows),
        "masks": sum(r["n_annotators"] for r in rows),
        "domain_counts": dict(counts),
        "fold_counts": {k: dict(v) for k, v in fold_counts.items()},
        "group_overlap_by_construction": 0,
        "errors": errors[:100],
    }
    (out / "riga_manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
