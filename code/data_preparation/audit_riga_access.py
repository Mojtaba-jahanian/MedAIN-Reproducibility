"""Validate the official RIGA+ record before downloading the full archive."""
from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED = {
    "name": "RIGAPlus.zip",
    "size": 1078033476,
    "checksum": "md5:4df43d5e8152776d32219dbb6b77d1e9",
}


def main(record_path: str) -> None:
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    files = {f["key"]: f for f in record.get("files", [])}
    item = files.get(EXPECTED["name"])
    checks = {
        "record_id": str(record.get("id")) == "6325549",
        "file_present": item is not None,
        "size_matches": bool(item and item.get("size") == EXPECTED["size"]),
        "checksum_matches": bool(item and item.get("checksum") == EXPECTED["checksum"]),
        "download_link_present": bool(item and item.get("links", {}).get("self")),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "doi": "10.5281/zenodo.6325549",
        "expected": EXPECTED,
        "checks": checks,
    }
    out = Path(__file__).resolve().parents[1] / "audit" / "riga_access_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])

