from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Callable, Dict, List, Set, Tuple
from urllib.error import URLError

from build_upcoming_odds import build_rows, resolve_date, write_csv
from get_data import DataConfig, LEAGUE_CODE_MAP, get_data
from live_odds import load_local_odds
from model import SimpleBettingModel, over_probability, poisson_pmf


BANKROLL = 1000.0
LEAGUES = sorted(LEAGUE_CODE_MAP.keys())
BLACKLIST = {
    "Real Valladolid",
    "Real Betis",
    "Athletic Club",
    "Rayo Vallecano",
    "Everton",
    "Crystal Palace",
    "Sheffield United",
    "Salernitana",
    "Empoli",
    "Darmstadt",
}
DEFAULT_MISSING_FIELDS = ["home_win_odds", "over_2_5_goals_odds", "over_9_5_corners_odds"]
PRO_LIVE_PROFILE = "pro_live"

LEAGUE_ALIASES = {
    "premierleague": "EPL",
    "epl": "EPL",
    "englandpremierleague": "EPL",
    "laliga": "LaLiga",
    "spanishlaliga": "LaLiga",
    "segunda": "LaLiga2",
    "bundesliga": "Bundesliga",
    "seriea": "SerieA",
    "ligue1": "Ligue1",
    "ligue2": "Ligue2",
    "portugal": "PrimeiraLiga",
    "primeiraliga": "PrimeiraLiga",
    "ered": "Eredivisie",
    "eredvisie": "Eredivisie",
    "mls": "USAMLS",
    "ligamx": "MexicoLigaMX",
    "brasilseriaa": "BrazilSerieA",
    "argentinaprimera": "ArgentinaPrimera",
}


def normalize_league_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        return ""
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    if compact in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[compact]
    for canonical in LEAGUES:
        if compact == re.sub(r"[^a-z0-9]", "", canonical.lower()):
            return canonical
    return name


def fair_odds_from_prob(p: float) -> float:
    if p <= 0:
        return math.inf
    return 1.0 / p


def edge_pct(book_odds: float, fair_odds: float) -> float:
    return ((book_odds / fair_odds) - 1.0) * 100.0


def kelly_fraction(odds: float, p: float) -> float:
    b = odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    k = (b * p - q) / b
    return max(0.0, k)


def under_probability(line: float, lam_total: float) -> float:
    return max(0.0, min(1.0, 1.0 - over_probability(line, lam_total)))


def btts_yes_probability(home_goal_lam: float, away_goal_lam: float) -> float:
    p_home_scores = 1.0 - math.exp(-home_goal_lam)
    p_away_scores = 1.0 - math.exp(-away_goal_lam)
    return max(0.0, min(1.0, p_home_scores * p_away_scores))


def btts_no_probability(home_goal_lam: float, away_goal_lam: float) -> float:
    return max(0.0, min(1.0, 1.0 - btts_yes_probability(home_goal_lam, away_goal_lam)))


def fmt_row(cols: List[str], widths: List[int]) -> str:
    out = []
    for i, c in enumerate(cols):
        out.append(c.ljust(widths[i]))
    return " | ".join(out)


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"


def should_use_color(color_mode: str) -> bool:
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def colorize_report_text(report_text: str, use_color: bool) -> str:
    if not use_color:
        return report_text
    out: List[str] = []
    confidence_col_idx = -1
    for line in report_text.splitlines():
        styled = line
        if " | " in line and "Confidence" in line:
            cols = line.split(" | ")
            confidence_col_idx = -1
            for i, c in enumerate(cols):
                if c.strip() == "Confidence":
                    confidence_col_idx = i
                    break
        elif line.strip() == "":
            confidence_col_idx = -1

        if confidence_col_idx >= 0 and " | " in line and "Confidence" not in line:
            cols = line.split(" | ")
            if len(cols) > confidence_col_idx and set(line.strip()) != {"-"}:
                raw = cols[confidence_col_idx].strip().replace("%", "")
                try:
                    conf = float(raw)
                    if conf >= 60.0:
                        c = ANSI_GREEN
                    elif conf < 50.0:
                        c = ANSI_RED
                    else:
                        c = ANSI_YELLOW
                    cols[confidence_col_idx] = f"{c}{cols[confidence_col_idx]}{ANSI_RESET}"
                    styled = " | ".join(cols)
                except ValueError:
                    pass

        if line.startswith("GLOBAL SNIPER SCANNER"):
            styled = f"{ANSI_BOLD}{ANSI_CYAN}{line}{ANSI_RESET}"
        elif line.startswith("Mode:"):
            styled = f"{ANSI_DIM}{line}{ANSI_RESET}"
        elif line.startswith("Scanning "):
            styled = f"{ANSI_YELLOW}{line}{ANSI_RESET}"
        elif line == "SNIPER REPORT":
            styled = f"{ANSI_BOLD}{ANSI_YELLOW}{line}{ANSI_RESET}"
        elif line.startswith("SNIPER REPORT (VALUE FOUND)"):
            styled = f"{ANSI_BOLD}{ANSI_GREEN}{line}{ANSI_RESET}"
        elif line.startswith("PARLAY SHORTLIST"):
            styled = f"{ANSI_BOLD}{ANSI_MAGENTA}{line}{ANSI_RESET}"
        elif line == "Corner Predictions":
            styled = f"{ANSI_BOLD}{ANSI_BLUE}{line}{ANSI_RESET}"
        elif line.startswith("No qualifying bets found"):
            styled = f"{ANSI_RED}{line}{ANSI_RESET}"
        out.append(styled)
    return "\n".join(out)


MarketFn = Callable[[dict], float]
MarketDef = Tuple[str, str, MarketFn]


def build_market_defs() -> List[MarketDef]:
    return [
        (
            "Home (Winner)",
            "home_win_odds",
            lambda p: match_outcome_probs(p["home_goal_lambda"], p["away_goal_lambda"])[0],
        ),
        (
            "Away (Winner)",
            "away_win_odds",
            lambda p: match_outcome_probs(p["home_goal_lambda"], p["away_goal_lambda"])[2],
        ),
        (
            "Home (DNB)",
            "home_dnb_odds",
            lambda p: home_dnb_probability(p["home_goal_lambda"], p["away_goal_lambda"]),
        ),
        (
            "Away (DNB)",
            "away_dnb_odds",
            lambda p: away_dnb_probability(p["home_goal_lambda"], p["away_goal_lambda"]),
        ),
        (
            "AH Home -0.5",
            "ah_home_m0_5_odds",
            lambda p: handicap_cover_probability(p["home_goal_lambda"], p["away_goal_lambda"], -0.5),
        ),
        (
            "AH Away +0.5",
            "ah_away_p0_5_odds",
            lambda p: handicap_cover_probability(p["away_goal_lambda"], p["home_goal_lambda"], +0.5),
        ),
        (
            "AH Home -1.5",
            "ah_home_m1_5_odds",
            lambda p: handicap_cover_probability(p["home_goal_lambda"], p["away_goal_lambda"], -1.5),
        ),
        (
            "AH Away +1.5",
            "ah_away_p1_5_odds",
            lambda p: handicap_cover_probability(p["away_goal_lambda"], p["home_goal_lambda"], +1.5),
        ),
        (
            "AH Corners Home -1.5",
            "ahc_home_m1_5_odds",
            lambda p: handicap_cover_probability(p["home_corner_lambda"], p["away_corner_lambda"], -1.5),
        ),
        (
            "AH Corners Away +1.5",
            "ahc_away_p1_5_odds",
            lambda p: handicap_cover_probability(p["away_corner_lambda"], p["home_corner_lambda"], +1.5),
        ),
        (
            "AH Corners Home -2.5",
            "ahc_home_m2_5_odds",
            lambda p: handicap_cover_probability(p["home_corner_lambda"], p["away_corner_lambda"], -2.5),
        ),
        (
            "AH Corners Away +2.5",
            "ahc_away_p2_5_odds",
            lambda p: handicap_cover_probability(p["away_corner_lambda"], p["home_corner_lambda"], +2.5),
        ),
        (
            "Over 1.5 Goals",
            "over_1_5_goals_odds",
            lambda p: over_probability(1.5, p["home_goal_lambda"] + p["away_goal_lambda"]),
        ),
        (
            "Over 2.5 Goals",
            "over_2_5_goals_odds",
            lambda p: over_probability(2.5, p["home_goal_lambda"] + p["away_goal_lambda"]),
        ),
        (
            "Over 3.5 Goals",
            "over_3_5_goals_odds",
            lambda p: over_probability(3.5, p["home_goal_lambda"] + p["away_goal_lambda"]),
        ),
        (
            "Under 2.5 Goals",
            "under_2_5_goals_odds",
            lambda p: under_probability(2.5, p["home_goal_lambda"] + p["away_goal_lambda"]),
        ),
        (
            "Under 3.5 Goals",
            "under_3_5_goals_odds",
            lambda p: under_probability(3.5, p["home_goal_lambda"] + p["away_goal_lambda"]),
        ),
        (
            "BTTS Yes",
            "btts_yes_odds",
            lambda p: btts_yes_probability(p["home_goal_lambda"], p["away_goal_lambda"]),
        ),
        (
            "BTTS No",
            "btts_no_odds",
            lambda p: btts_no_probability(p["home_goal_lambda"], p["away_goal_lambda"]),
        ),
        (
            "Over 8.5 Corners",
            "over_8_5_corners_odds",
            lambda p: over_probability(8.5, p["home_corner_lambda"] + p["away_corner_lambda"]),
        ),
        (
            "Over 9.5 Corners",
            "over_9_5_corners_odds",
            lambda p: over_probability(9.5, p["home_corner_lambda"] + p["away_corner_lambda"]),
        ),
        (
            "Over 10.5 Corners",
            "over_10_5_corners_odds",
            lambda p: over_probability(10.5, p["home_corner_lambda"] + p["away_corner_lambda"]),
        ),
        (
            "Under 9.5 Corners",
            "under_9_5_corners_odds",
            lambda p: under_probability(9.5, p["home_corner_lambda"] + p["away_corner_lambda"]),
        ),
        (
            "Under 10.5 Corners",
            "under_10_5_corners_odds",
            lambda p: under_probability(10.5, p["home_corner_lambda"] + p["away_corner_lambda"]),
        ),
    ]


def match_outcome_probs(home_lam: float, away_lam: float, max_goals: int = 10) -> Tuple[float, float, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    p_home_tail = 0.0
    p_away_tail = 0.0

    for h in range(max_goals + 1):
        p_h = poisson_pmf(h, home_lam)
        for a in range(max_goals + 1):
            p = p_h * poisson_pmf(a, away_lam)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
        p_home_tail += p_h

    for a in range(max_goals + 1):
        p_away_tail += poisson_pmf(a, away_lam)

    # Renormalize because we truncate goal space at max_goals.
    mass = max(1e-9, p_home_tail * p_away_tail)
    home_win /= mass
    draw /= mass
    away_win /= mass
    return home_win, draw, away_win


def home_dnb_probability(home_lam: float, away_lam: float) -> float:
    home_win, draw, _ = match_outcome_probs(home_lam, away_lam)
    active_mass = max(1e-9, 1.0 - draw)
    return max(0.0, min(1.0, home_win / active_mass))


def away_dnb_probability(home_lam: float, away_lam: float) -> float:
    _, draw, away_win = match_outcome_probs(home_lam, away_lam)
    active_mass = max(1e-9, 1.0 - draw)
    return max(0.0, min(1.0, away_win / active_mass))


def fair_odds_home_dnb(home_lam: float, away_lam: float) -> float:
    home_win, draw, _ = match_outcome_probs(home_lam, away_lam)
    if home_win <= 0:
        return math.inf
    return max(1.01, (1.0 - draw) / home_win)


def fair_odds_away_dnb(home_lam: float, away_lam: float) -> float:
    _, draw, away_win = match_outcome_probs(home_lam, away_lam)
    if away_win <= 0:
        return math.inf
    return max(1.01, (1.0 - draw) / away_win)


def handicap_cover_probability(
    home_lam: float, away_lam: float, handicap: float, max_goals: int = 10
) -> float:
    p = 0.0
    total = 0.0
    for h in range(max_goals + 1):
        p_h = poisson_pmf(h, home_lam)
        for a in range(max_goals + 1):
            joint = p_h * poisson_pmf(a, away_lam)
            total += joint
            if (h + handicap) > a:
                p += joint
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, p / total))


def print_table(lines: List[List[str]]) -> str:
    if not lines:
        return ""
    widths = [len(h) for h in lines[0]]
    for r in lines[1:]:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    rendered = []
    rendered.append(fmt_row(lines[0], widths))
    rendered.append("-" * (sum(widths) + (3 * (len(widths) - 1))))
    for r in lines[1:]:
        rendered.append(fmt_row(r, widths))
    return "\n".join(rendered)


def conflict_group_for_market(market_name: str) -> str:
    m = market_name.lower().strip()
    if m.startswith("btts "):
        return "btts"
    if "goals" in m:
        if "1.5" in m:
            return "goals_1.5"
        if "2.5" in m:
            return "goals_2.5"
        if "3.5" in m:
            return "goals_3.5"
    if "corners" in m:
        if "8.5" in m:
            return "corners_8.5"
        if "9.5" in m:
            return "corners_9.5"
        if "10.5" in m:
            return "corners_10.5"
    return ""


def market_key_from_name(market_name: str) -> str:
    m = market_name.lower().strip()
    mapping = {
        "home (winner)": "home",
        "away (winner)": "away",
        "home (dnb)": "dnb",
        "away (dnb)": "adnb",
        "ah home -0.5": "ah_h_m0_5",
        "ah away +0.5": "ah_a_p0_5",
        "ah home -1.5": "ah_h_m1_5",
        "ah away +1.5": "ah_a_p1_5",
        "ah corners home -1.5": "ahc_h_m1_5",
        "ah corners away +1.5": "ahc_a_p1_5",
        "ah corners home -2.5": "ahc_h_m2_5",
        "ah corners away +2.5": "ahc_a_p2_5",
        "over 1.5 goals": "o15",
        "over 2.5 goals": "o25",
        "over 3.5 goals": "o35",
        "under 2.5 goals": "u25",
        "under 3.5 goals": "u35",
        "btts yes": "btts_yes",
        "btts no": "btts_no",
        "over 8.5 corners": "co85",
        "over 9.5 corners": "co95",
        "over 10.5 corners": "co105",
        "under 9.5 corners": "cu95",
        "under 10.5 corners": "cu105",
    }
    return mapping.get(m, "")


def parse_market_keys(raw: str) -> Set[str]:
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _field_has_odds(row: dict, field: str) -> bool:
    try:
        return float(row.get(field, 0.0)) > 1.01
    except (TypeError, ValueError):
        return False


def _row_key(date_val: str, league: str, home: str, away: str) -> Tuple[str, str, str, str]:
    return ((date_val or "").strip(), (league or "").strip(), (home or "").strip(), (away or "").strip())


def _raw_live_to_bool(raw: str) -> bool:
    v = (raw or "").strip().lower()
    return v in {"1", "true", "yes", "y", "live", "inplay", "in-play"}


def _filter_rows_by_date(rows: List[dict], target_date: str, all_games: bool, live_only: bool) -> Tuple[List[dict], str]:
    filtered: List[dict] = []
    for row in rows:
        if live_only and not row.get("is_live", False):
            continue
        if not all_games and (not row["date"] or row["date"] != target_date):
            continue
        filtered.append(row)
    effective_date = target_date
    if not all_games and not filtered:
        dated_rows = [r for r in rows if r.get("date")]
        future_or_today = sorted({r["date"] for r in dated_rows if r["date"] >= target_date})
        if future_or_today:
            effective_date = future_or_today[0]
            filtered = [r for r in rows if r.get("date") == effective_date and (not live_only or r.get("is_live", False))]
    return filtered, effective_date


def show_missing_odds(
    odds_csv: str,
    target_date: str,
    all_games: bool,
    live_only: bool,
    fields: List[str],
    export_path: str,
) -> int:
    rows = load_local_odds(odds_csv)
    for row in rows:
        row["league"] = normalize_league_name(row.get("league", ""))

    filtered_rows, effective_date = _filter_rows_by_date(rows, target_date, all_games, live_only)
    if not filtered_rows:
        print("No fixtures found for selected date filter.")
        return 0

    table = [["Date", "League", "Fixture", "Missing Odds Fields"]]
    export_rows: List[dict] = []
    for r in filtered_rows:
        missing = [f for f in fields if not _field_has_odds(r, f)]
        if not missing:
            continue
        table.append(
            [
                r.get("date") or "N/A",
                r.get("league") or "N/A",
                f'{r.get("home_team", "")} vs {r.get("away_team", "")}',
                ", ".join(missing),
            ]
        )
        out = {
            "date": r.get("date", ""),
            "league": r.get("league", ""),
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
        }
        for f in fields:
            out[f] = ""
        export_rows.append(out)

    print(f"Missing-odds check date: {effective_date}")
    if len(table) == 1:
        print("All filtered fixtures have required odds fields.")
        return 0
    print(print_table(table))

    if export_path:
        cols = ["date", "league", "home_team", "away_team"] + fields
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(export_rows)
        print(f"Exported fill template: {export_path}")
    return 0


def merge_filled_odds(odds_csv: str, filled_csv: str) -> int:
    with open(odds_csv, newline="", encoding="utf-8") as f:
        base_reader = csv.DictReader(f)
        if not base_reader.fieldnames:
            print(f"Invalid base CSV header: {odds_csv}")
            return 1
        base_fields = base_reader.fieldnames
        base_rows = list(base_reader)

    with open(filled_csv, newline="", encoding="utf-8") as f:
        fill_reader = csv.DictReader(f)
        if not fill_reader.fieldnames:
            print(f"Invalid filled CSV header: {filled_csv}")
            return 1
        fill_rows = list(fill_reader)

    key_cols = ("date", "league", "home_team", "away_team")
    fill_map = {}
    for r in fill_rows:
        key = tuple((r.get(c) or "").strip() for c in key_cols)
        if any(not k for k in key):
            continue
        fill_map[key] = r

    updated_cells = 0
    updated_rows = 0
    for r in base_rows:
        key = tuple((r.get(c) or "").strip() for c in key_cols)
        fr = fill_map.get(key)
        if not fr:
            continue
        row_changed = False
        for col, val in fr.items():
            if col in key_cols:
                continue
            if col not in base_fields:
                continue
            new_val = (val or "").strip()
            if not new_val:
                continue
            if r.get(col, "") != new_val:
                r[col] = new_val
                updated_cells += 1
                row_changed = True
        if row_changed:
            updated_rows += 1

    with open(odds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(base_rows)

    print(f"Merged filled odds from {filled_csv} into {odds_csv}")
    print(f"Updated rows: {updated_rows}, updated cells: {updated_cells}")
    return 0


def auto_fill_odds(
    odds_csv: str,
    season_code: str,
    local_data_dir: str,
    target_date: str,
    all_games: bool,
    live_only: bool,
    margin_pct: float,
    overwrite: bool,
) -> int:
    try:
        history = get_data(DataConfig(leagues=LEAGUES, season_code=season_code, local_data_dir=local_data_dir))
    except (URLError, ValueError) as e:
        print(f"Could not load model history for auto-fill: {e}")
        return 1

    history.extend([{**r, "League": "GLOBAL"} for r in history])
    model = SimpleBettingModel()
    model.fit(history)

    parsed_rows = load_local_odds(odds_csv)
    for row in parsed_rows:
        row["league"] = normalize_league_name(row.get("league", ""))
    filtered_rows, effective_date = _filter_rows_by_date(parsed_rows, target_date, all_games, live_only)
    target_keys: Set[Tuple[str, str, str, str]] = set(
        _row_key(r.get("date", ""), r.get("league", ""), r.get("home_team", ""), r.get("away_team", ""))
        for r in filtered_rows
    )
    if not target_keys:
        print("No fixtures selected for auto-fill.")
        return 0

    with open(odds_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Invalid CSV header in {odds_csv}")
            return 1
        fields = reader.fieldnames
        raw_rows = list(reader)

    calculators: Dict[str, Callable[[dict], float]] = {
        "home_win_odds": lambda p: fair_odds_from_prob(
            match_outcome_probs(p["home_goal_lambda"], p["away_goal_lambda"])[0]
        ),
        "away_win_odds": lambda p: fair_odds_from_prob(
            match_outcome_probs(p["home_goal_lambda"], p["away_goal_lambda"])[2]
        ),
        "home_dnb_odds": lambda p: fair_odds_home_dnb(p["home_goal_lambda"], p["away_goal_lambda"]),
        "away_dnb_odds": lambda p: fair_odds_away_dnb(p["home_goal_lambda"], p["away_goal_lambda"]),
        "ah_home_m0_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["home_goal_lambda"], p["away_goal_lambda"], -0.5)
        ),
        "ah_away_p0_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["away_goal_lambda"], p["home_goal_lambda"], +0.5)
        ),
        "ah_home_m1_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["home_goal_lambda"], p["away_goal_lambda"], -1.5)
        ),
        "ah_away_p1_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["away_goal_lambda"], p["home_goal_lambda"], +1.5)
        ),
        "ahc_home_m1_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["home_corner_lambda"], p["away_corner_lambda"], -1.5)
        ),
        "ahc_away_p1_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["away_corner_lambda"], p["home_corner_lambda"], +1.5)
        ),
        "ahc_home_m2_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["home_corner_lambda"], p["away_corner_lambda"], -2.5)
        ),
        "ahc_away_p2_5_odds": lambda p: fair_odds_from_prob(
            handicap_cover_probability(p["away_corner_lambda"], p["home_corner_lambda"], +2.5)
        ),
        "over_1_5_goals_odds": lambda p: fair_odds_from_prob(
            over_probability(1.5, p["home_goal_lambda"] + p["away_goal_lambda"])
        ),
        "over_2_5_goals_odds": lambda p: fair_odds_from_prob(
            over_probability(2.5, p["home_goal_lambda"] + p["away_goal_lambda"])
        ),
        "over_3_5_goals_odds": lambda p: fair_odds_from_prob(
            over_probability(3.5, p["home_goal_lambda"] + p["away_goal_lambda"])
        ),
        "under_2_5_goals_odds": lambda p: fair_odds_from_prob(
            under_probability(2.5, p["home_goal_lambda"] + p["away_goal_lambda"])
        ),
        "under_3_5_goals_odds": lambda p: fair_odds_from_prob(
            under_probability(3.5, p["home_goal_lambda"] + p["away_goal_lambda"])
        ),
        "btts_yes_odds": lambda p: fair_odds_from_prob(
            btts_yes_probability(p["home_goal_lambda"], p["away_goal_lambda"])
        ),
        "btts_no_odds": lambda p: fair_odds_from_prob(
            btts_no_probability(p["home_goal_lambda"], p["away_goal_lambda"])
        ),
        "over_8_5_corners_odds": lambda p: fair_odds_from_prob(
            over_probability(8.5, p["home_corner_lambda"] + p["away_corner_lambda"])
        ),
        "over_9_5_corners_odds": lambda p: fair_odds_from_prob(
            over_probability(9.5, p["home_corner_lambda"] + p["away_corner_lambda"])
        ),
        "over_10_5_corners_odds": lambda p: fair_odds_from_prob(
            over_probability(10.5, p["home_corner_lambda"] + p["away_corner_lambda"])
        ),
        "under_9_5_corners_odds": lambda p: fair_odds_from_prob(
            under_probability(9.5, p["home_corner_lambda"] + p["away_corner_lambda"])
        ),
        "under_10_5_corners_odds": lambda p: fair_odds_from_prob(
            under_probability(10.5, p["home_corner_lambda"] + p["away_corner_lambda"])
        ),
    }

    updated_rows = 0
    updated_cells = 0
    margin_factor = max(0.01, 1.0 - (margin_pct / 100.0))

    for r in raw_rows:
        league_norm = normalize_league_name(r.get("league", ""))
        key = _row_key(r.get("date", ""), league_norm, r.get("home_team", ""), r.get("away_team", ""))
        if key not in target_keys:
            continue
        if live_only and not _raw_live_to_bool(r.get("is_live", "")) and not _raw_live_to_bool(r.get("status", "")):
            continue

        pred_league = league_norm if league_norm in model.params_by_league else "GLOBAL"
        pred = model.predict(pred_league, (r.get("home_team") or "").strip(), (r.get("away_team") or "").strip())

        row_changed = False
        for col, fair_fn in calculators.items():
            if col not in fields:
                continue
            existing = (r.get(col) or "").strip()
            if not overwrite:
                try:
                    if float(existing) > 1.01:
                        continue
                except (TypeError, ValueError):
                    pass
            fair = fair_fn(pred)
            if not math.isfinite(fair) or fair <= 1.01:
                continue
            synthetic_book = max(1.02, fair * margin_factor)
            new_val = f"{synthetic_book:.2f}"
            if existing != new_val:
                r[col] = new_val
                updated_cells += 1
                row_changed = True
        if row_changed:
            updated_rows += 1

    with open(odds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    print(
        f"Auto-filled missing odds for {updated_rows} fixtures ({updated_cells} cells) "
        f"on date filter {effective_date} into {odds_csv}"
    )
    print(f"Synthetic margin used: {margin_pct:.1f}%")
    return 0


def mode_settings(mode: str) -> Tuple[float, float, float]:
    # min_edge, max_risk_pct, kelly_multiplier
    if mode == "safe":
        return 10.0, 2.0, 0.33
    if mode == "aggressive":
        return 3.0, 6.0, 0.75
    return 5.0, 3.0, 0.35


def profile_settings(profile: str) -> Dict[str, object]:
    if profile == PRO_LIVE_PROFILE:
        return {
            "allowed_market_keys": {"o25"},
            "default_min_edge": 9.0,
            "default_odds_min": 2.0,
            "default_odds_max": 3.2,
            "force_one_pick_per_match": True,
            "flat_stake_pct": 0.25,
            "day_stop_loss_units": -3.0,
            "total_stop_loss_units": -20.0,
            "bets_log_csv": "bets_log.csv",
        }
    return {
        "allowed_market_keys": set(),
        "default_min_edge": None,
        "default_odds_min": None,
        "default_odds_max": None,
        "force_one_pick_per_match": False,
        "flat_stake_pct": 0.0,
        "day_stop_loss_units": None,
        "total_stop_loss_units": None,
        "bets_log_csv": "bets_log.csv",
    }


def read_log_pnl_units(bets_log_csv: str, target_date: str) -> Tuple[float, float]:
    if not os.path.exists(bets_log_csv):
        return 0.0, 0.0
    day_pnl = 0.0
    total_pnl = 0.0
    with open(bets_log_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            settled = (r.get("settled") or "").strip().lower()
            if settled not in {"1", "true", "yes"}:
                continue
            try:
                pnl_u = float(r.get("pnl_units", "0") or 0.0)
            except (TypeError, ValueError):
                continue
            total_pnl += pnl_u
            if (r.get("date") or "").strip() == target_date:
                day_pnl += pnl_u
    return day_pnl, total_pnl


def append_bets_log(bets_log_csv: str, picks: List[dict], run_ts: str, profile: str) -> int:
    if not picks:
        return 0
    existing_open = set()
    existing_rows = []
    if os.path.exists(bets_log_csv):
        with open(bets_log_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            for r in existing_rows:
                settled = (r.get("settled") or "").strip().lower()
                if settled in {"1", "true", "yes"}:
                    continue
                key = (
                    (r.get("date") or "").strip(),
                    (r.get("league") or "").strip(),
                    (r.get("fixture") or "").strip(),
                    (r.get("market") or "").strip(),
                )
                existing_open.add(key)

    fields = [
        "logged_at",
        "profile",
        "date",
        "league",
        "fixture",
        "market",
        "book_odds",
        "fair_odds",
        "edge_pct",
        "confidence_pct",
        "stake",
        "stake_units",
        "result",
        "settled",
        "closing_odds",
        "clv_pct",
        "pnl_units",
    ]

    to_write = []
    for p in picks:
        key = (p["date"], p["league"], p["fixture"], p["market"])
        if key in existing_open:
            continue
        to_write.append(
            {
                "logged_at": run_ts,
                "profile": profile,
                "date": p["date"],
                "league": p["league"],
                "fixture": p["fixture"],
                "market": p["market"],
                "book_odds": f"{p['book']:.2f}",
                "fair_odds": f"{p['fair']:.2f}",
                "edge_pct": f"{p['edge']:.2f}",
                "confidence_pct": f"{p['confidence']:.2f}",
                "stake": f"{p['stake']:.2f}",
                "stake_units": f"{p['stake_units']:.2f}",
                "result": "",
                "settled": "0",
                "closing_odds": "",
                "clv_pct": "",
                "pnl_units": "",
            }
        )

    if not to_write:
        return 0

    file_exists = os.path.exists(bets_log_csv)
    with open(bets_log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerows(to_write)
    return len(to_write)


def run_scan(
    season_code: str,
    odds_csv: str,
    min_edge: float | None,
    local_data_dir: str,
    target_date: str,
    all_games: bool,
    live_only: bool,
    mode: str,
    max_picks: int,
    one_pick_per_match: bool,
    odds_min: float,
    odds_max: float,
    profile: str,
    bankroll: float,
    bets_log_csv: str,
    include_markets: Set[str],
    exclude_markets: Set[str],
    shortlist_market_cap: int,
    color_mode: str,
    log_picks: bool,
) -> int:
    default_min_edge, max_risk_pct, kelly_multiplier = mode_settings(mode)
    profile_cfg = profile_settings(profile)
    profile_min_edge = profile_cfg["default_min_edge"]
    profile_odds_min = profile_cfg["default_odds_min"]
    profile_odds_max = profile_cfg["default_odds_max"]
    force_one_pick_per_match = bool(profile_cfg["force_one_pick_per_match"])
    allowed_market_keys = set(profile_cfg["allowed_market_keys"])
    flat_stake_pct = float(profile_cfg["flat_stake_pct"])
    day_stop_loss_units = profile_cfg["day_stop_loss_units"]
    total_stop_loss_units = profile_cfg["total_stop_loss_units"]
    effective_min_edge = (
        profile_min_edge
        if min_edge is None and profile_min_edge is not None
        else (default_min_edge if min_edge is None else min_edge)
    )
    effective_odds_min = profile_odds_min if odds_min <= 0 and profile_odds_min is not None else odds_min
    effective_odds_max = profile_odds_max if odds_max <= 0 and profile_odds_max is not None else odds_max
    effective_one_pick_per_match = one_pick_per_match or force_one_pick_per_match
    use_color = should_use_color(color_mode)

    try:
        history = get_data(DataConfig(leagues=LEAGUES, season_code=season_code, local_data_dir=local_data_dir))
    except (URLError, ValueError) as e:
        print("Could not load league history (network/DNS issue or missing local files).")
        print(
            "Use --local-data-dir with historical CSVs named by league code "
            "(E0, E1, E2, E3, EC, SP1, SP2, D1, D2, I1, I2, F1, F2, N1, P1, B1, T1, SC0, A1, DNK, SWE, NOR, etc)."
        )
        print(f"Details: {e}")
        return 1
    history.extend([{**r, "League": "GLOBAL"} for r in history])
    model = SimpleBettingModel()
    model.fit(history)
    odds_rows = load_local_odds(odds_csv)
    for row in odds_rows:
        row["league_raw"] = row.get("league", "")
        row["league"] = normalize_league_name(row.get("league", ""))
    filtered_rows: List[dict] = []
    for row in odds_rows:
        if live_only and not row.get("is_live", False):
            continue
        if not all_games and (not row["date"] or row["date"] != target_date):
            continue
        filtered_rows.append(row)
    effective_date = target_date
    if not all_games and not filtered_rows:
        dated_rows = [r for r in odds_rows if r.get("date")]
        future_or_today = sorted({r["date"] for r in dated_rows if r["date"] >= target_date})
        if future_or_today:
            effective_date = future_or_today[0]
            filtered_rows = [r for r in odds_rows if r.get("date") == effective_date]

    out_lines: List[str] = []
    out_lines.append("GLOBAL SNIPER SCANNER (V3.1 KEYLESS)")
    out_lines.append(
        f"Mode: {mode} | Min edge: {effective_min_edge:.1f}% | Max risk: {max_risk_pct:.1f}%"
    )
    if profile:
        out_lines.append(f"Profile: {profile}")
    if effective_odds_min > 0 or effective_odds_max > 0:
        out_lines.append(f"Odds filter: min={effective_odds_min:.2f}, max={effective_odds_max:.2f}")
    if live_only:
        out_lines.append("Mode: LIVE only (rows with is_live=1/status=live)")
    if not all_games:
        if effective_date != target_date:
            out_lines.append(
                f"No fixtures found for {target_date}. Auto-switched to next available date: {effective_date}."
            )
        else:
            out_lines.append(f"Fixture date: {effective_date}")
    out_lines.append("")

    if profile == PRO_LIVE_PROFILE:
        stop_date = target_date if target_date != "today" else date.today().isoformat()
        day_pnl_u, total_pnl_u = read_log_pnl_units(bets_log_csv, stop_date)
        if (
            (day_stop_loss_units is not None and day_pnl_u <= float(day_stop_loss_units))
            or (total_stop_loss_units is not None and total_pnl_u <= float(total_stop_loss_units))
        ):
            out_lines.append("LIVE RISK GUARD")
            out_lines.append(
                f"Stop-loss triggered (day={day_pnl_u:.2f}u, total={total_pnl_u:.2f}u). No new bets generated."
            )
            report_text = "\n".join(out_lines)
            print(colorize_report_text(report_text, use_color))
            return 0

    leagues_in_scan = []
    seen_leagues = set()
    for r in filtered_rows:
        l = r["league"]
        if l in seen_leagues:
            continue
        seen_leagues.add(l)
        leagues_in_scan.append(l)

    for league in leagues_in_scan:
        out_lines.append(f"Scanning {league}...")
        league_rows = [
            r
            for r in filtered_rows
            if r["league"] == league
            and r["home_team"] not in BLACKLIST
            and r["away_team"] not in BLACKLIST
        ]
        if not league_rows:
            out_lines.append("  No fixtures in odds CSV for selected date.")
            out_lines.append("")
            continue
        sched = [["Date", "Match"]]
        for r in league_rows:
            minute_txt = f" [{r['minute']}']" if live_only and r.get("minute", 0) > 0 else ""
            sched.append([r.get("date") or "N/A", f'{r["home_team"]} vs {r["away_team"]}{minute_txt}'])
        out_lines.append(print_table(sched))
        out_lines.append("")

    header = ["Date", "League", "Fixture", "Market", "Book", "Fair", "Confidence", "Edge", "Stake"]
    rows: List[List[str]] = []
    rows_for_sort: List[Tuple[float, float, str, str, str, str, List[str]]] = []
    corner_lines: List[str] = []
    market_defs = build_market_defs()

    for row in filtered_rows:
        league = row["league"]
        home = row["home_team"]
        away = row["away_team"]
        if home in BLACKLIST or away in BLACKLIST:
            continue

        pred_league = league if league in model.params_by_league else "GLOBAL"
        pred = model.predict(pred_league, home, away)
        expected_corners = pred["home_corner_lambda"] + pred["away_corner_lambda"]
        if expected_corners >= 9.2:
            corner_lines.append(
                f"{league} | {home} vs {away}: Expect Over 9.5 ({expected_corners:.1f})"
            )

        markets = []
        for market_name, odds_field, prob_fn in market_defs:
            market_key = market_key_from_name(market_name)
            if include_markets and market_key not in include_markets:
                continue
            if exclude_markets and market_key in exclude_markets:
                continue
            if allowed_market_keys and market_key not in allowed_market_keys:
                continue
            book = float(row.get(odds_field, 0.0))
            if book <= 1.01:
                continue
            if effective_odds_min > 0 and book < effective_odds_min:
                continue
            if effective_odds_max > 0 and book > effective_odds_max:
                continue
            p = prob_fn(pred)
            if odds_field == "home_dnb_odds":
                fair = fair_odds_home_dnb(pred["home_goal_lambda"], pred["away_goal_lambda"])
            elif odds_field == "away_dnb_odds":
                fair = fair_odds_away_dnb(pred["home_goal_lambda"], pred["away_goal_lambda"])
            else:
                fair = fair_odds_from_prob(p)
            markets.append((market_name, p, book, fair))

        for market_name, p, book, fair in markets:
            edge = edge_pct(book, fair)
            if edge < effective_min_edge:
                continue
            k = kelly_fraction(book, p)
            if flat_stake_pct > 0:
                stake = bankroll * (flat_stake_pct / 100.0)
            else:
                stake = min(bankroll * (max_risk_pct / 100.0), bankroll * k * kelly_multiplier)
            if stake <= 0:
                continue
            fixture = f"{home} vs {away}"
            confidence_pct = p * 100.0
            rows_for_sort.append(
                (
                    edge,
                    confidence_pct,
                    row["date"] or "N/A",
                    league,
                    fixture,
                    conflict_group_for_market(market_name),
                    [
                        row["date"] or "N/A",
                        league,
                        fixture,
                        market_name,
                        f"{book:.2f}",
                        f"{fair:.2f}",
                        f"{confidence_pct:.1f}%",
                        f"+{edge:.1f}%",
                        f"${stake:.2f}",
                    ],
                )
            )

    rows_for_sort.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Avoid contradictory picks on the same match (for example over/under or BTTS yes/no pairs).
    seen_conflict_groups: Set[Tuple[str, str, str, str]] = set()
    filtered_ranked: List[Tuple[float, float, str, str, str, str, List[str]]] = []
    for item in rows_for_sort:
        _, _, d, lg, fx, conflict_group, _ = item
        if conflict_group:
            conflict_key = (d, lg, fx, conflict_group)
            if conflict_key in seen_conflict_groups:
                continue
            seen_conflict_groups.add(conflict_key)
        filtered_ranked.append(item)

    for _, _, _, _, _, _, r in filtered_ranked:
        rows.append(r)

    if rows:
        out_lines.append("SNIPER REPORT (VALUE FOUND)")
        out_lines.append(print_table([header] + rows))
    else:
        out_lines.append("SNIPER REPORT")
        out_lines.append("No qualifying bets found for the selected date.")

    shortlist_picks: List[dict] = []
    if rows and max_picks > 0:
        shortlist: List[List[str]] = []
        seen_matches: Set[Tuple[str, str, str]] = set()
        market_counts: Dict[str, int] = {}
        for edge, confidence, d, lg, fx, _, r in filtered_ranked:
            if effective_one_pick_per_match:
                match_key = (d, lg, fx)
                if match_key in seen_matches:
                    continue
                seen_matches.add(match_key)
            if shortlist_market_cap > 0:
                mk = market_key_from_name(r[3])
                current = market_counts.get(mk, 0)
                if current >= shortlist_market_cap:
                    continue
                market_counts[mk] = current + 1
            shortlist.append(r)
            try:
                book_val = float(r[4])
                fair_val = float(r[5])
                stake_val = float(r[8].replace("$", "").strip())
            except (TypeError, ValueError):
                book_val = 0.0
                fair_val = 0.0
                stake_val = 0.0
            unit_size = bankroll * (flat_stake_pct / 100.0) if flat_stake_pct > 0 else 1.0
            stake_units = stake_val / unit_size if unit_size > 0 else 0.0
            shortlist_picks.append(
                {
                    "date": d,
                    "league": lg,
                    "fixture": fx,
                    "market": r[3],
                    "book": book_val,
                    "fair": fair_val,
                    "edge": edge,
                    "confidence": confidence,
                    "stake": stake_val,
                    "stake_units": stake_units,
                }
            )
            if len(shortlist) >= max_picks:
                break

        if shortlist:
            out_lines.append("")
            out_lines.append(f"PARLAY SHORTLIST (TOP {len(shortlist)})")
            out_lines.append(print_table([header] + shortlist))

    out_lines.append("")
    out_lines.append("Corner Predictions")
    if corner_lines:
        out_lines.extend(corner_lines)
    else:
        out_lines.append("No strong corner projections today.")

    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join("reports", f"sniper_report_{ts}.txt")
    report_text = "\n".join(out_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(colorize_report_text(report_text, use_color))
    print("")
    print(f"Report saved to: {report_path}")
    if (profile == PRO_LIVE_PROFILE or log_picks) and shortlist_picks:
        logged = append_bets_log(
            bets_log_csv=bets_log_csv,
            picks=shortlist_picks,
            run_ts=datetime.now().isoformat(timespec="seconds"),
            profile=profile if profile else "manual",
        )
        if logged > 0:
            print(f"Logged {logged} picks to {bets_log_csv}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyless soccer value betting scanner")
    parser.add_argument("--season-code", default="2526", help="football-data season code")
    parser.add_argument("--odds-csv", default="upcoming_odds.csv", help="Local odds CSV path")
    parser.add_argument("--min-edge", type=float, default=None, help="Minimum edge percent (overrides mode)")
    parser.add_argument(
        "--mode",
        choices=["safe", "balanced", "aggressive"],
        default="balanced",
        help="Risk profile presets for edge threshold and staking.",
    )
    parser.add_argument(
        "--local-data-dir",
        default="",
        help="Optional folder with historical CSVs named by league code (E0.csv, SP1.csv, ...)",
    )
    parser.add_argument(
        "--date",
        default="today",
        help="Fixture date to scan: today, tomorrow, or YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Scan all fixtures in the odds CSV (disables --date filter)",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Only scan in-play fixtures (requires CSV column is_live or status).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously rescan odds CSV at an interval.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between scans when --watch is enabled.",
    )
    parser.add_argument(
        "--refresh-odds",
        action="store_true",
        help="Rebuild odds CSV from free fixture feeds before scanning.",
    )
    parser.add_argument(
        "--refresh-date",
        default="",
        help="Date used for --refresh-odds (today, tomorrow, or YYYY-MM-DD). Default follows --date.",
    )
    parser.add_argument(
        "--refresh-lookahead-days",
        type=int,
        default=3,
        help="When refreshing odds, if date has no fixtures, search forward this many days.",
    )
    parser.add_argument(
        "--show-missing-odds",
        action="store_true",
        help="List fixtures missing required odds fields and exit.",
    )
    parser.add_argument(
        "--missing-fields",
        default=",".join(DEFAULT_MISSING_FIELDS),
        help="Comma-separated odds fields required for --show-missing-odds.",
    )
    parser.add_argument(
        "--export-missing-template",
        default="",
        help="Optional CSV path to export fixtures that need odds.",
    )
    parser.add_argument(
        "--merge-filled-odds",
        default="",
        help="CSV path of filled odds template to merge back into --odds-csv, then exit.",
    )
    parser.add_argument(
        "--auto-fill-odds",
        action="store_true",
        help="Auto-fill missing odds in --odds-csv using model-derived synthetic prices.",
    )
    parser.add_argument(
        "--auto-fill-margin",
        type=float,
        default=6.0,
        help="Bookmaker margin percent used when auto-filling synthetic odds.",
    )
    parser.add_argument(
        "--auto-fill-overwrite",
        action="store_true",
        help="Overwrite existing odds values when using --auto-fill-odds.",
    )
    parser.add_argument(
        "--max-picks",
        type=int,
        default=0,
        help="Size of bottom parlay shortlist (0 = hide shortlist).",
    )
    parser.add_argument(
        "--one-pick-per-match",
        action="store_true",
        help="For parlay shortlist only: keep top-ranked pick per fixture.",
    )
    parser.add_argument("--odds-min", type=float, default=0.0, help="Minimum book odds filter (0 disables).")
    parser.add_argument("--odds-max", type=float, default=0.0, help="Maximum book odds filter (0 disables).")
    parser.add_argument(
        "--include-markets",
        default="",
        help="Optional comma-separated market keys to include only.",
    )
    parser.add_argument(
        "--exclude-markets",
        default="",
        help="Optional comma-separated market keys to exclude.",
    )
    parser.add_argument(
        "--shortlist-market-cap",
        type=int,
        default=0,
        help="Optional max picks per market key in parlay shortlist (0 disables).",
    )
    parser.add_argument(
        "--profile",
        choices=["none", PRO_LIVE_PROFILE],
        default="none",
        help="Optional execution profile. Use pro_live for strict controlled-live settings.",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=BANKROLL,
        help="Bankroll used for stake sizing.",
    )
    parser.add_argument(
        "--bets-log-csv",
        default="bets_log.csv",
        help="Path to bet log CSV (used by pro_live stop-loss and logging).",
    )
    parser.add_argument(
        "--log-picks",
        action="store_true",
        help="Log parlay shortlist picks to --bets-log-csv even without a profile.",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Terminal color mode for scanner output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_profile = "" if args.profile == "none" else args.profile
    scan_date = resolve_date(args.date)
    missing_fields = [f.strip() for f in args.missing_fields.split(",") if f.strip()]
    include_markets = parse_market_keys(args.include_markets)
    exclude_markets = parse_market_keys(args.exclude_markets)
    if args.refresh_odds:
        refresh_raw = args.refresh_date if args.refresh_date else args.date
        refresh_date = resolve_date(refresh_raw)
        try:
            built_rows, effective_refresh_date = build_rows(
                args.season_code,
                refresh_date,
                lookahead_days=max(0, args.refresh_lookahead_days),
            )
            write_csv(args.odds_csv, built_rows)
            has_book_odds = any(float((r.get("home_win_odds") or "0") or 0) > 1.01 for r in built_rows)
            if effective_refresh_date != refresh_date:
                print(
                    f"Refreshed odds CSV with {len(built_rows)} fixtures for {effective_refresh_date} "
                    f"(requested {refresh_date}): {args.odds_csv}"
                )
            else:
                print(f"Refreshed odds CSV with {len(built_rows)} fixtures for {refresh_date}: {args.odds_csv}")
            if not has_book_odds:
                print("Note: fixture-only refresh (no bookmaker odds found). Add odds manually for value picks.")
        except ValueError as e:
            print(f"Could not refresh odds CSV: {e}")
            raise SystemExit(1)
    if args.merge_filled_odds:
        raise SystemExit(merge_filled_odds(args.odds_csv, args.merge_filled_odds))
    if args.auto_fill_odds:
        raise SystemExit(
            auto_fill_odds(
                odds_csv=args.odds_csv,
                season_code=args.season_code,
                local_data_dir=args.local_data_dir,
                target_date=scan_date,
                all_games=args.all_games,
                live_only=args.live_only,
                margin_pct=args.auto_fill_margin,
                overwrite=args.auto_fill_overwrite,
            )
        )
    if args.show_missing_odds:
        raise SystemExit(
            show_missing_odds(
                odds_csv=args.odds_csv,
                target_date=scan_date,
                all_games=args.all_games,
                live_only=args.live_only,
                fields=missing_fields,
                export_path=args.export_missing_template,
            )
        )
    if args.watch:
        while True:
            print("")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Live rescan...")
            run_scan(
                args.season_code,
                args.odds_csv,
                args.min_edge,
                args.local_data_dir,
                scan_date,
                args.all_games,
                args.live_only,
                args.mode,
                args.max_picks,
                args.one_pick_per_match,
                args.odds_min,
                args.odds_max,
                selected_profile,
                args.bankroll,
                args.bets_log_csv,
                include_markets,
                exclude_markets,
                max(0, args.shortlist_market_cap),
                args.color,
                args.log_picks,
            )
            time.sleep(max(5, args.interval))
    raise SystemExit(
        run_scan(
            args.season_code,
            args.odds_csv,
            args.min_edge,
            args.local_data_dir,
            scan_date,
            args.all_games,
            args.live_only,
            args.mode,
            args.max_picks,
            args.one_pick_per_match,
            args.odds_min,
            args.odds_max,
            selected_profile,
            args.bankroll,
            args.bets_log_csv,
            include_markets,
            exclude_markets,
            max(0, args.shortlist_market_cap),
            args.color,
            args.log_picks,
        )
    )
