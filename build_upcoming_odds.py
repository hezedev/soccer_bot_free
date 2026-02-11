from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from get_data import LEAGUE_CODE_MAP


OUT_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "is_live",
    "minute",
    "home_win_odds",
    "away_win_odds",
    "home_dnb_odds",
    "away_dnb_odds",
    "ah_home_m0_5_odds",
    "ah_away_p0_5_odds",
    "ah_home_m1_5_odds",
    "ah_away_p1_5_odds",
    "ahc_home_m1_5_odds",
    "ahc_away_p1_5_odds",
    "ahc_home_m2_5_odds",
    "ahc_away_p2_5_odds",
    "over_1_5_goals_odds",
    "over_2_5_goals_odds",
    "over_3_5_goals_odds",
    "under_2_5_goals_odds",
    "under_3_5_goals_odds",
    "btts_yes_odds",
    "btts_no_odds",
    "over_8_5_corners_odds",
    "over_9_5_corners_odds",
    "over_10_5_corners_odds",
    "under_9_5_corners_odds",
    "under_10_5_corners_odds",
]

SOURCE_PRIORITY: Dict[str, List[str]] = {
    "home_win_odds": ["B365H", "PSH", "WHH", "AvgH", "MaxH"],
    "over_2_5_goals_odds": ["B365>2.5", "P>2.5", "Avg>2.5", "Max>2.5"],
    "under_2_5_goals_odds": ["B365<2.5", "P<2.5", "Avg<2.5", "Max<2.5"],
    "btts_yes_odds": ["B365BTS", "PSBTS", "AvgBTS", "MaxBTS"],
    "btts_no_odds": ["B365BTNS", "PSBTNS", "AvgBTNS", "MaxBTNS"],
}

ESPN_LEAGUE_MAP: Dict[str, str] = {
    "EPL": "eng.1",
    "Championship": "eng.2",
    "LeagueOne": "eng.3",
    "LeagueTwo": "eng.4",
    "LaLiga": "esp.1",
    "LaLiga2": "esp.2",
    "Bundesliga": "ger.1",
    "Bundesliga2": "ger.2",
    "SerieA": "ita.1",
    "SerieB": "ita.2",
    "Ligue1": "fra.1",
    "Ligue2": "fra.2",
    "Eredivisie": "ned.1",
    "PrimeiraLiga": "por.1",
    "BelgiumProLeague": "bel.1",
    "TurkeySuperLig": "tur.1",
    "ScotlandPremiership": "sco.1",
    "AustriaBundesliga": "aut.1",
    "DenmarkSuperliga": "den.1",
    "SwedenAllsvenskan": "swe.1",
    "NorwayEliteserien": "nor.1",
    "USAMLS": "usa.1",
    "MexicoLigaMX": "mex.1",
    "BrazilSerieA": "bra.1",
    "ArgentinaPrimera": "arg.1",
}


def _build_url(league_code: str, season_code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"


def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _odds_value(row: dict, choices: List[str]) -> str:
    for col in choices:
        v = (row.get(col) or "").strip()
        if not v:
            continue
        try:
            f = float(v)
            if f > 1.01:
                return f"{f:.2f}"
        except ValueError:
            continue
    return ""


def _read_csv_from_url(url: str) -> List[dict]:
    with urlopen(url, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _read_json_from_url(url: str) -> dict:
    with urlopen(url, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return json.loads(text)


def resolve_date(raw: str) -> str:
    v = raw.strip().lower()
    today = date.today()
    if v == "today":
        return today.isoformat()
    if v == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    return raw


def _build_rows_exact_date(season_code: str, target_date: str) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    failed: List[str] = []
    for league, code in LEAGUE_CODE_MAP.items():
        try:
            feed_rows = _read_csv_from_url(_build_url(code, season_code))
        except (HTTPError, URLError, TimeoutError, OSError):
            failed.append(league)
            continue

        for r in feed_rows:
            match_date = _parse_date(r.get("Date", ""))
            if match_date != target_date:
                continue
            home = (r.get("HomeTeam") or "").strip()
            away = (r.get("AwayTeam") or "").strip()
            if not home or not away:
                continue

            out = {c: "" for c in OUT_COLUMNS}
            out["date"] = match_date
            out["league"] = league
            out["home_team"] = home
            out["away_team"] = away
            out["is_live"] = "0"
            out["minute"] = "0"

            for out_col, src_cols in SOURCE_PRIORITY.items():
                out[out_col] = _odds_value(r, src_cols)

            rows.append(out)
    return rows, failed


def _build_rows_espn_date(target_date: str) -> List[dict]:
    ymd = target_date.replace("-", "")
    rows: List[dict] = []
    seen = set()
    for league_name, espn_code in ESPN_LEAGUE_MAP.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard?dates={ymd}"
        try:
            payload = _read_json_from_url(url)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
            continue
        for ev in payload.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            competitors = comps[0].get("competitors", [])
            if len(competitors) < 2:
                continue

            home_name = ""
            away_name = ""
            for c in competitors:
                team_name = (
                    (c.get("team", {}) or {}).get("displayName")
                    or (c.get("team", {}) or {}).get("shortDisplayName")
                    or ""
                ).strip()
                if c.get("homeAway") == "home":
                    home_name = team_name
                elif c.get("homeAway") == "away":
                    away_name = team_name

            if not home_name or not away_name:
                home_name = (
                    (competitors[0].get("team", {}) or {}).get("displayName", "").strip()
                )
                away_name = (
                    (competitors[1].get("team", {}) or {}).get("displayName", "").strip()
                )
            if not home_name or not away_name:
                continue

            key = (target_date, league_name, home_name.lower(), away_name.lower())
            if key in seen:
                continue
            seen.add(key)

            out = {c: "" for c in OUT_COLUMNS}
            out["date"] = target_date
            out["league"] = league_name
            out["home_team"] = home_name
            out["away_team"] = away_name
            out["is_live"] = "0"
            out["minute"] = "0"
            rows.append(out)
    rows.sort(key=lambda x: (x["league"], x["home_team"], x["away_team"]))
    return rows


def build_rows(season_code: str, target_date: str, lookahead_days: int = 0) -> Tuple[List[dict], str]:
    first_failed: List[str] = []
    start = datetime.strptime(target_date, "%Y-%m-%d").date()
    max_days = max(0, lookahead_days)
    for offset in range(max_days + 1):
        d = (start + timedelta(days=offset)).isoformat()
        rows, failed = _build_rows_exact_date(season_code, d)
        if offset == 0:
            first_failed = failed
        if rows:
            rows.sort(key=lambda x: (x["league"], x["home_team"], x["away_team"]))
            return rows, d
        fixture_rows = _build_rows_espn_date(d)
        if fixture_rows:
            return fixture_rows, d

    details = f" No fixtures found from {target_date} to {(start + timedelta(days=max_days)).isoformat()}."
    if first_failed:
        details += f" Failed leagues: {', '.join(first_failed)}"
    raise ValueError("Could not build upcoming odds." + details)


def write_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build upcoming_odds.csv from free feeds")
    parser.add_argument("--season-code", default="2526", help="football-data season code")
    parser.add_argument("--date", default="tomorrow", help="today, tomorrow, or YYYY-MM-DD")
    parser.add_argument("--out", default="upcoming_odds.csv", help="Output CSV path")
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=0,
        help="If no fixtures on --date, try next N days until fixtures are found.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_date = resolve_date(args.date)
    built, effective_date = build_rows(args.season_code, target_date, lookahead_days=args.lookahead_days)
    write_csv(args.out, built)
    print(f"Built {len(built)} fixtures for {effective_date} -> {args.out}")
