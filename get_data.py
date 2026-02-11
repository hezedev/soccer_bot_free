from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


LEAGUE_CODE_MAP: Dict[str, str] = {
    "EPL": "E0",
    "Championship": "E1",
    "LeagueOne": "E2",
    "LeagueTwo": "E3",
    "NationalLeague": "EC",
    "LaLiga": "SP1",
    "LaLiga2": "SP2",
    "Bundesliga": "D1",
    "Bundesliga2": "D2",
    "SerieA": "I1",
    "SerieB": "I2",
    "Ligue1": "F1",
    "Ligue2": "F2",
    "Eredivisie": "N1",
    "PrimeiraLiga": "P1",
    "BelgiumProLeague": "B1",
    "TurkeySuperLig": "T1",
    "ScotlandPremiership": "SC0",
    "AustriaBundesliga": "A1",
    "DenmarkSuperliga": "DNK",
    "SwedenAllsvenskan": "SWE",
    "NorwayEliteserien": "NOR",
    "GreeceSuperLeague": "G1",
    "SwitzerlandSuperLeague": "SWZ",
    "PolandEkstraklasa": "POL",
    "CzechFirstLeague": "CZ",
    "RomaniaLigaI": "ROU",
    "CroatiaHNL": "CRO",
    "BulgariaFirstLeague": "BUL",
    "SlovakiaSuperLiga": "SLK",
    "SloveniaPrvaLiga": "SVN",
    "HungaryNB1": "HUN",
    "SerbiaSuperLiga": "SRB",
    "UkrainePremierLeague": "UKR",
    "RussiaPremierLeague": "RUS",
    "FinlandVeikkausliiga": "FIN",
    "IrelandPremierDivision": "IRL",
    "BrazilSerieA": "BRA",
    "ArgentinaPrimera": "ARG",
    "USAMLS": "USA",
    "MexicoLigaMX": "MEX",
    "JapanJ1": "JPN",
    "ChinaSuperLeague": "CHN",
}


@dataclass
class DataConfig:
    leagues: List[str]
    season_code: str = "2526"
    local_data_dir: str = ""


def _build_url(league_code: str, season_code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"


def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _to_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _read_csv_text(path_or_url: str, is_url: bool) -> str:
    if is_url:
        with urlopen(path_or_url, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    with open(path_or_url, encoding="utf-8", newline="") as f:
        return f.read()


def get_data(config: DataConfig) -> List[dict]:
    rows: List[dict] = []
    failed_sources: List[str] = []

    for league_name in config.leagues:
        code = LEAGUE_CODE_MAP.get(league_name)
        if not code:
            continue
        local_path = ""
        if config.local_data_dir:
            local_path = os.path.join(config.local_data_dir, f"{code}.csv")
        try:
            if local_path and os.path.exists(local_path):
                text = _read_csv_text(local_path, is_url=False)
            else:
                url = _build_url(code, config.season_code)
                text = _read_csv_text(url, is_url=True)
        except (HTTPError, URLError, TimeoutError, OSError):
            failed_sources.append(league_name)
            continue
        reader = csv.DictReader(io.StringIO(text))
        for r in reader:
            home = (r.get("HomeTeam") or "").strip()
            away = (r.get("AwayTeam") or "").strip()
            if not home or not away:
                continue
            fthg = _to_float(r.get("FTHG"), default=-1.0)
            ftag = _to_float(r.get("FTAG"), default=-1.0)
            if fthg < 0 or ftag < 0:
                continue
            rows.append(
                {
                    "League": league_name,
                    "Date": _parse_date(r.get("Date", "")),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": fthg,
                    "FTAG": ftag,
                    "HC": _to_float(r.get("HC"), default=0.0),
                    "AC": _to_float(r.get("AC"), default=0.0),
                }
            )

    if not rows:
        msg = "No data downloaded. Check internet access/season_code/local_data_dir."
        if failed_sources:
            msg += f" Failed leagues: {', '.join(failed_sources)}"
        raise ValueError(msg)
    return rows
