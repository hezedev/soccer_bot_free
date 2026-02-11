from __future__ import annotations

import argparse
import csv
import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from get_data import DataConfig, LEAGUE_CODE_MAP, get_data


def resolve_date(raw: str) -> str:
    token = (raw or "").strip().lower()
    today = date.today()
    if token == "today":
        return today.isoformat()
    if token == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    if token == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    return (raw or "").strip()


def to_float(raw: str, default: float = 0.0) -> float:
    try:
        return float((raw or "").strip())
    except (TypeError, ValueError):
        return default


def is_settled(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "y"}


def outcome_home_away(home: int, away: int) -> str:
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "draw"


def settle_market(market: str, home_goals: int, away_goals: int, home_corners: Optional[int], away_corners: Optional[int]) -> str:
    m = (market or "").strip()
    m_low = m.lower()
    goals_total = home_goals + away_goals
    corners_total = (home_corners + away_corners) if (home_corners is not None and away_corners is not None) else None

    if m_low == "home (winner)":
        return "win" if home_goals > away_goals else "loss"
    if m_low == "away (winner)":
        return "win" if away_goals > home_goals else "loss"
    if m_low == "home (dnb)":
        out = outcome_home_away(home_goals, away_goals)
        return "push" if out == "draw" else ("win" if out == "home" else "loss")
    if m_low == "away (dnb)":
        out = outcome_home_away(home_goals, away_goals)
        return "push" if out == "draw" else ("win" if out == "away" else "loss")

    # Generic AH goals: "AH Home -0.5", "AH Away +1.5"
    ah_match = re.match(r"^ah\s+(home|away)\s+([+-]\d+(?:\.\d+)?)$", m_low)
    if ah_match:
        side = ah_match.group(1)
        handicap = float(ah_match.group(2))
        diff = (home_goals - away_goals) if side == "home" else (away_goals - home_goals)
        adj = diff + handicap
        if adj > 0:
            return "win"
        if adj < 0:
            return "loss"
        return "push"

    # Generic AH corners: "AH Corners Home -1.5", "AH Corners Away +2.5"
    ahc_match = re.match(r"^ah\s+corners\s+(home|away)\s+([+-]\d+(?:\.\d+)?)$", m_low)
    if ahc_match:
        if home_corners is None or away_corners is None:
            return "pending"
        side = ahc_match.group(1)
        handicap = float(ahc_match.group(2))
        diff = (home_corners - away_corners) if side == "home" else (away_corners - home_corners)
        adj = diff + handicap
        if adj > 0:
            return "win"
        if adj < 0:
            return "loss"
        return "push"

    over_under = re.match(r"^(over|under)\s+(\d+(?:\.\d+)?)\s+(goals|corners)$", m_low)
    if over_under:
        side = over_under.group(1)
        line = float(over_under.group(2))
        kind = over_under.group(3)
        total = goals_total if kind == "goals" else corners_total
        if total is None:
            return "pending"
        if side == "over":
            if total > line:
                return "win"
            if total < line:
                return "loss"
            return "push"
        if total < line:
            return "win"
        if total > line:
            return "loss"
        return "push"

    if m_low == "btts yes":
        return "win" if (home_goals > 0 and away_goals > 0) else "loss"
    if m_low == "btts no":
        return "win" if (home_goals == 0 or away_goals == 0) else "loss"

    return "pending"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-settle bets_log.csv from historical match results")
    p.add_argument("--bets-log-csv", default="bets_log.csv", help="Path to bets log CSV")
    p.add_argument("--season-code", default="2526", help="football-data season code")
    p.add_argument(
        "--local-data-dir",
        default="",
        help="Optional folder with historical CSVs named by league code (E0.csv, SP1.csv, ...)",
    )
    p.add_argument("--up-to-date", default="yesterday", help="Settle bets with date <= this (today/yesterday/YYYY-MM-DD)")
    p.add_argument("--profile", default="", help="Optional profile filter, e.g. pro_live")
    return p.parse_args()


def run() -> int:
    args = parse_args()
    cutoff = resolve_date(args.up_to_date)

    with open(args.bets_log_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Invalid CSV header in {args.bets_log_csv}")
            return 1
        fields = reader.fieldnames
        rows = list(reader)

    leagues = sorted(LEAGUE_CODE_MAP.keys())
    try:
        data_rows = get_data(DataConfig(leagues=leagues, season_code=args.season_code, local_data_dir=args.local_data_dir))
    except ValueError as e:
        print(f"Could not load results data: {e}")
        return 1

    idx: Dict[Tuple[str, str, str, str], dict] = {}
    for r in data_rows:
        key = ((r.get("Date") or "").strip(), (r.get("League") or "").strip(), (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip())
        idx[key] = r

    updated = 0
    settled_now = 0
    still_pending = 0
    won = 0
    lost = 0
    pushed = 0

    for r in rows:
        if is_settled(r.get("settled", "")):
            continue
        if args.profile and (r.get("profile") or "").strip() != args.profile.strip():
            continue
        bet_date = (r.get("date") or "").strip()
        if not bet_date or (cutoff and bet_date > cutoff):
            continue

        fixture = (r.get("fixture") or "").strip()
        if " vs " not in fixture:
            still_pending += 1
            continue
        home, away = [x.strip() for x in fixture.split(" vs ", 1)]
        key = (bet_date, (r.get("league") or "").strip(), home, away)
        match = idx.get(key)
        if not match:
            still_pending += 1
            continue

        hg = int(to_float(str(match.get("FTHG", "0"))))
        ag = int(to_float(str(match.get("FTAG", "0"))))
        hc_raw = match.get("HC", None)
        ac_raw = match.get("AC", None)
        hc = int(to_float(str(hc_raw))) if hc_raw not in (None, "") else None
        ac = int(to_float(str(ac_raw))) if ac_raw not in (None, "") else None

        status = settle_market((r.get("market") or "").strip(), hg, ag, hc, ac)
        if status == "pending":
            still_pending += 1
            continue

        stake_units = to_float(r.get("stake_units", "0"))
        book_odds = to_float(r.get("book_odds", "0"))
        if status == "win":
            pnl_u = stake_units * max(0.0, (book_odds - 1.0))
            won += 1
        elif status == "loss":
            pnl_u = -stake_units
            lost += 1
        else:
            pnl_u = 0.0
            pushed += 1

        r["result"] = status
        r["settled"] = "1"
        r["pnl_units"] = f"{pnl_u:.2f}"
        updated += 1
        settled_now += 1

    with open(args.bets_log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Settlement complete up to {cutoff}.")
    print(f"Updated rows: {updated}")
    print(f"Settled now: {settled_now} | Won: {won} | Lost: {lost} | Push: {pushed} | Pending unresolved: {still_pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
