from __future__ import annotations

import csv
from datetime import datetime
from typing import List


BASE_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
]


def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(raw: str) -> bool:
    if raw is None:
        return False
    v = str(raw).strip().lower()
    return v in {"1", "true", "yes", "y", "live", "inplay", "in-play"}


def _to_int(raw: str, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def load_local_odds(csv_path: str = "upcoming_odds.csv") -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")
        missing = [c for c in BASE_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        rows: List[dict] = []
        for r in reader:
            out = {
                "date": _parse_date(r.get("date", "")),
                "league": (r.get("league") or "").strip(),
                "home_team": (r.get("home_team") or "").strip(),
                "away_team": (r.get("away_team") or "").strip(),
                "is_live": _to_bool(r.get("is_live", "")) or _to_bool(r.get("status", "")),
                "minute": _to_int(r.get("minute", ""), default=0),
            }
            for field in reader.fieldnames:
                if field in BASE_COLUMNS:
                    continue
                if field.endswith("_odds"):
                    out[field] = _to_float(r.get(field))
            rows.append(out)
    return rows
