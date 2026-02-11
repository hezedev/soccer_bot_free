from __future__ import annotations

import argparse
import csv
import io
import math
from datetime import datetime
from itertools import product
from statistics import pstdev
from typing import Dict, List, Set, Tuple
from urllib.request import urlopen

from get_data import LEAGUE_CODE_MAP
from model import SimpleBettingModel, over_probability, poisson_pmf


MARKET_KEYS = [
    "home",
    "dnb",
    "o15",
    "o25",
    "o35",
    "u25",
    "u35",
    "btts_yes",
    "btts_no",
    "co85",
    "co95",
    "co105",
    "cu95",
    "cu105",
]


def _build_url(league_code: str, season_code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"


def _read_csv(url: str) -> List[dict]:
    with urlopen(url, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


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


def _to_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _first_odds(row: dict, cols: List[str]) -> float:
    for c in cols:
        v = _to_float(row.get(c), 0.0)
        if v > 1.01:
            return v
    return 0.0


def fair_odds(p: float) -> float:
    if p <= 0:
        return 1e9
    return 1.0 / p


def edge_pct(book: float, fair: float) -> float:
    return ((book / fair) - 1.0) * 100.0


def win_home(row: dict) -> bool:
    return _to_float(row.get("FTHG"), -1) > _to_float(row.get("FTAG"), -1)


def win_over25(row: dict) -> bool:
    return (_to_float(row.get("FTHG"), 0) + _to_float(row.get("FTAG"), 0)) > 2.5


def total_goals(row: dict) -> float:
    return _to_float(row.get("FTHG"), 0) + _to_float(row.get("FTAG"), 0)


def total_corners(row: dict) -> float:
    return _to_float(row.get("HC"), 0) + _to_float(row.get("AC"), 0)


def btts_yes(row: dict) -> bool:
    return _to_float(row.get("FTHG"), 0) > 0 and _to_float(row.get("FTAG"), 0) > 0


def home_win_probability(home_lam: float, away_lam: float, max_goals: int = 10) -> float:
    p = 0.0
    for h in range(max_goals + 1):
        p_h = poisson_pmf(h, home_lam)
        for a in range(max_goals + 1):
            if h > a:
                p += p_h * poisson_pmf(a, away_lam)
    return max(0.0, min(1.0, p))


def match_outcome_probs(home_lam: float, away_lam: float, max_goals: int = 10) -> Tuple[float, float, float]:
    home = 0.0
    draw = 0.0
    away = 0.0
    for h in range(max_goals + 1):
        p_h = poisson_pmf(h, home_lam)
        for a in range(max_goals + 1):
            p = p_h * poisson_pmf(a, away_lam)
            if h > a:
                home += p
            elif h == a:
                draw += p
            else:
                away += p
    mass = max(1e-9, home + draw + away)
    return home / mass, draw / mass, away / mass


def under_probability(line: float, lam_total: float) -> float:
    return max(0.0, min(1.0, 1.0 - over_probability(line, lam_total)))


def btts_yes_probability(home_lam: float, away_lam: float) -> float:
    p_home = 1.0 - math.exp(-home_lam)
    p_away = 1.0 - math.exp(-away_lam)
    return max(0.0, min(1.0, p_home * p_away))


def btts_no_probability(home_lam: float, away_lam: float) -> float:
    return max(0.0, min(1.0, 1.0 - btts_yes_probability(home_lam, away_lam)))


def home_dnb_probability(home_lam: float, away_lam: float) -> float:
    p_home, p_draw, _ = match_outcome_probs(home_lam, away_lam)
    active = max(1e-9, 1.0 - p_draw)
    return max(0.0, min(1.0, p_home / active))


def expand_markets(markets: List[str]) -> List[str]:
    out: List[str] = []
    for m in markets:
        key = m.strip().lower()
        if not key:
            continue
        if key == "all":
            out.extend(
                [
                    "home",
                    "dnb",
                    "o15",
                    "o25",
                    "o35",
                    "u25",
                    "u35",
                    "btts_yes",
                    "btts_no",
                    "co85",
                    "co95",
                    "co105",
                    "cu95",
                    "cu105",
                ]
            )
        else:
            out.append(key)
    seen = set()
    dedup = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return dedup


def _parse_csv_set(raw: str) -> Set[str]:
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _parse_weekday_set(raw: str) -> Set[int]:
    if not raw:
        return set()
    items = {x.strip().lower() for x in raw.split(",") if x.strip()}
    name_to_idx = {
        "mon": 0,
        "monday": 0,
        "tue": 1,
        "tues": 1,
        "tuesday": 1,
        "wed": 2,
        "wednesday": 2,
        "thu": 3,
        "thur": 3,
        "thurs": 3,
        "thursday": 3,
        "fri": 4,
        "friday": 4,
        "sat": 5,
        "saturday": 5,
        "sun": 6,
        "sunday": 6,
    }
    out = set()
    for it in items:
        if it.isdigit():
            v = int(it)
            if 0 <= v <= 6:
                out.add(v)
            continue
        if it in name_to_idx:
            out.add(name_to_idx[it])
    return out


def _weekday_from_date_iso(date_iso: str) -> int:
    try:
        return datetime.fromisoformat(date_iso).weekday()
    except Exception:
        return -1


def _load_rows(season_code: str) -> List[dict]:
    raw_rows: List[dict] = []
    for league, code in LEAGUE_CODE_MAP.items():
        try:
            rows = _read_csv(_build_url(code, season_code))
        except Exception:
            continue
        for r in rows:
            home = (r.get("HomeTeam") or "").strip()
            away = (r.get("AwayTeam") or "").strip()
            if not home or not away:
                continue
            fthg = _to_float(r.get("FTHG"), -1)
            ftag = _to_float(r.get("FTAG"), -1)
            if fthg < 0 or ftag < 0:
                continue
            row = {
                "League": league,
                "Date": _parse_date(r.get("Date", "")),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
                "HC": _to_float(r.get("HC"), 0.0),
                "AC": _to_float(r.get("AC"), 0.0),
            }
            raw_rows.append({**r, **row})
    raw_rows.sort(key=lambda x: (x.get("Date", ""), x.get("League", ""), x.get("HomeTeam", "")))
    return raw_rows


def evaluate_backtest(
    rows: List[dict],
    season_code: str,
    train_ratio: float,
    min_edge: float,
    markets: List[str],
    odds_min: float,
    odds_max: float,
    include_leagues: Set[str],
    exclude_leagues: Set[str],
    include_weekdays: Set[int],
    exclude_weekdays: Set[int],
) -> Dict[str, object]:
    if not rows:
        return {"error": "No historical rows downloaded."}

    split_idx = max(1, int(len(rows) * train_ratio))
    train = rows[:split_idx]
    test = rows[split_idx:]
    if not test:
        return {"error": "No test rows after split; lower train_ratio."}

    model = SimpleBettingModel()
    model.fit(train)

    # market_name, bookmaker columns, win_fn, probability_fn
    defs = []
    if "home" in markets:
        defs.append(
            (
                "HomeWin",
                ["B365H", "PSH", "WHH", "AvgH", "MaxH"],
                win_home,
                lambda p: p["home_win_prob"],
            )
        )
    if "dnb" in markets:
        defs.append(
            (
                "HomeDNB",
                ["B365DNBH", "PSDNBH", "AvgDNBH", "MaxDNBH"],
                win_home,
                lambda p: p["home_dnb_prob"],
            )
        )
    if "o15" in markets:
        defs.append(
            (
                "Over1.5",
                ["B365>1.5", "P>1.5", "Avg>1.5", "Max>1.5"],
                lambda r: total_goals(r) > 1.5,
                lambda p: p["over15_prob"],
            )
        )
    if "o25" in markets:
        defs.append(
            (
                "Over2.5",
                ["B365>2.5", "P>2.5", "Avg>2.5", "Max>2.5"],
                win_over25,
                lambda p: p["over25_prob"],
            )
        )
    if "o35" in markets:
        defs.append(
            (
                "Over3.5",
                ["B365>3.5", "P>3.5", "Avg>3.5", "Max>3.5"],
                lambda r: total_goals(r) > 3.5,
                lambda p: p["over35_prob"],
            )
        )
    if "u25" in markets:
        defs.append(
            (
                "Under2.5",
                ["B365<2.5", "P<2.5", "Avg<2.5", "Max<2.5"],
                lambda r: total_goals(r) < 2.5,
                lambda p: p["under25_prob"],
            )
        )
    if "u35" in markets:
        defs.append(
            (
                "Under3.5",
                ["B365<3.5", "P<3.5", "Avg<3.5", "Max<3.5"],
                lambda r: total_goals(r) < 3.5,
                lambda p: p["under35_prob"],
            )
        )
    if "btts_yes" in markets:
        defs.append(
            (
                "BTTSYes",
                ["B365BTS", "PSBTS", "AvgBTS", "MaxBTS"],
                btts_yes,
                lambda p: p["btts_yes_prob"],
            )
        )
    if "btts_no" in markets:
        defs.append(
            (
                "BTTSNo",
                ["B365BTNS", "PSBTNS", "AvgBTNS", "MaxBTNS"],
                lambda r: not btts_yes(r),
                lambda p: p["btts_no_prob"],
            )
        )
    if "co85" in markets:
        defs.append(
            (
                "CornersOver8.5",
                ["B365C>8.5", "PSC>8.5", "AvgC>8.5", "MaxC>8.5"],
                lambda r: total_corners(r) > 8.5,
                lambda p: p["co85_prob"],
            )
        )
    if "co95" in markets:
        defs.append(
            (
                "CornersOver9.5",
                ["B365C>9.5", "PSC>9.5", "AvgC>9.5", "MaxC>9.5"],
                lambda r: total_corners(r) > 9.5,
                lambda p: p["co95_prob"],
            )
        )
    if "co105" in markets:
        defs.append(
            (
                "CornersOver10.5",
                ["B365C>10.5", "PSC>10.5", "AvgC>10.5", "MaxC>10.5"],
                lambda r: total_corners(r) > 10.5,
                lambda p: p["co105_prob"],
            )
        )
    if "cu95" in markets:
        defs.append(
            (
                "CornersUnder9.5",
                ["B365C<9.5", "PSC<9.5", "AvgC<9.5", "MaxC<9.5"],
                lambda r: total_corners(r) < 9.5,
                lambda p: p["cu95_prob"],
            )
        )
    if "cu105" in markets:
        defs.append(
            (
                "CornersUnder10.5",
                ["B365C<10.5", "PSC<10.5", "AvgC<10.5", "MaxC<10.5"],
                lambda r: total_corners(r) < 10.5,
                lambda p: p["cu105_prob"],
            )
        )

    bets = 0
    wins = 0
    pnl = 0.0
    by_market: Dict[str, Dict[str, float]] = {}

    for r in test:
        league = r["League"]
        if include_leagues and league not in include_leagues:
            continue
        if exclude_leagues and league in exclude_leagues:
            continue
        wd = _weekday_from_date_iso(r.get("Date", ""))
        if include_weekdays and wd not in include_weekdays:
            continue
        if exclude_weekdays and wd in exclude_weekdays:
            continue

        pred_league = league if league in model.params_by_league else None
        if not pred_league:
            continue
        pred = model.predict(pred_league, r["HomeTeam"], r["AwayTeam"])
        total_goal_lam = pred["home_goal_lambda"] + pred["away_goal_lambda"]
        total_corner_lam = pred["home_corner_lambda"] + pred["away_corner_lambda"]
        probs = {
            "home_win_prob": home_win_probability(pred["home_goal_lambda"], pred["away_goal_lambda"]),
            "home_dnb_prob": home_dnb_probability(pred["home_goal_lambda"], pred["away_goal_lambda"]),
            "over15_prob": over_probability(1.5, total_goal_lam),
            "over25_prob": over_probability(2.5, total_goal_lam),
            "over35_prob": over_probability(3.5, total_goal_lam),
            "under25_prob": under_probability(2.5, total_goal_lam),
            "under35_prob": under_probability(3.5, total_goal_lam),
            "btts_yes_prob": btts_yes_probability(pred["home_goal_lambda"], pred["away_goal_lambda"]),
            "btts_no_prob": btts_no_probability(pred["home_goal_lambda"], pred["away_goal_lambda"]),
            "co85_prob": over_probability(8.5, total_corner_lam),
            "co95_prob": over_probability(9.5, total_corner_lam),
            "co105_prob": over_probability(10.5, total_corner_lam),
            "cu95_prob": under_probability(9.5, total_corner_lam),
            "cu105_prob": under_probability(10.5, total_corner_lam),
        }

        for market_name, odd_cols, win_fn, prob_fn in defs:
            book = _first_odds(r, odd_cols)
            if book <= 1.01:
                continue
            if odds_min > 0 and book < odds_min:
                continue
            if odds_max > 0 and book > odds_max:
                continue
            p = prob_fn(probs)
            fair = fair_odds(p)
            e = edge_pct(book, fair)
            if e < min_edge:
                continue

            won = win_fn(r)
            unit_pnl = (book - 1.0) if won else -1.0
            bets += 1
            wins += 1 if won else 0
            pnl += unit_pnl

            mk = by_market.setdefault(market_name, {"bets": 0, "wins": 0, "pnl": 0.0})
            mk["bets"] += 1
            mk["wins"] += 1 if won else 0
            mk["pnl"] += unit_pnl

    if bets == 0:
        return {
            "season_code": season_code,
            "train_ratio": train_ratio,
            "bets": 0,
            "wins": 0,
            "hit_rate": 0.0,
            "pnl": 0.0,
            "roi": 0.0,
            "by_market": by_market,
        }

    roi = (pnl / bets) * 100.0
    hit = (wins / bets) * 100.0
    return {
        "season_code": season_code,
        "train_ratio": train_ratio,
        "bets": bets,
        "wins": wins,
        "hit_rate": hit,
        "pnl": pnl,
        "roi": roi,
        "by_market": by_market,
    }


def print_backtest_result(result: Dict[str, object]) -> int:
    if "error" in result:
        print(result["error"])
        return 1
    bets = int(result["bets"])
    if bets == 0:
        print("No qualifying bets in test period. Try lower --min-edge or loosen filters.")
        return 0

    print(f"Backtest season: {result['season_code']}")
    tr = float(result["train_ratio"])
    print(f"Split: train {tr:.0%} / test {1-tr:.0%}")
    print(
        f"Bets: {result['bets']} | Wins: {result['wins']} | Hit rate: {result['hit_rate']:.1f}% "
        f"| PnL (1u flat): {result['pnl']:.2f}u | ROI: {result['roi']:.2f}%"
    )
    print("")
    print("By market:")
    by_market = result["by_market"]
    for k, v in sorted(by_market.items()):
        b = int(v["bets"])
        w = int(v["wins"])
        r = (v["pnl"] / b) * 100.0 if b else 0.0
        h = (w / b) * 100.0 if b else 0.0
        print(f"- {k}: bets={b}, hit={h:.1f}%, pnl={v['pnl']:.2f}u, roi={r:.2f}%")
    return 0


def run_grid_search(
    rows: List[dict],
    season_code: str,
    train_ratio: float,
    markets: List[str],
    min_edges: List[float],
    odds_mins: List[float],
    odds_maxs: List[float],
    include_leagues: Set[str],
    exclude_leagues: Set[str],
    include_weekdays: Set[int],
    exclude_weekdays: Set[int],
    min_bets: int,
    top_n: int,
) -> int:
    results: List[Dict[str, object]] = []
    for min_edge, odds_min, odds_max in product(min_edges, odds_mins, odds_maxs):
        if odds_max > 0 and odds_min > 0 and odds_min > odds_max:
            continue
        res = evaluate_backtest(
            rows=rows,
            season_code=season_code,
            train_ratio=train_ratio,
            min_edge=min_edge,
            markets=markets,
            odds_min=odds_min,
            odds_max=odds_max,
            include_leagues=include_leagues,
            exclude_leagues=exclude_leagues,
            include_weekdays=include_weekdays,
            exclude_weekdays=exclude_weekdays,
        )
        if "error" in res:
            print(res["error"])
            return 1
        if int(res["bets"]) < min_bets:
            continue
        res["cfg_min_edge"] = min_edge
        res["cfg_odds_min"] = odds_min
        res["cfg_odds_max"] = odds_max
        results.append(res)

    if not results:
        print("No grid configurations met min-bets threshold.")
        return 0

    results.sort(key=lambda r: (float(r["roi"]), float(r["pnl"]), int(r["bets"])), reverse=True)
    k = min(top_n, len(results))
    print(f"Grid Search Top {k}")
    print("min_edge | odds_min | odds_max | bets | hit_rate | pnl_u | roi")
    print("----------------------------------------------------------------")
    for r in results[:k]:
        print(
            f"{r['cfg_min_edge']:.1f}     | {r['cfg_odds_min']:.2f}     | {r['cfg_odds_max']:.2f}     | "
            f"{int(r['bets'])}   | {float(r['hit_rate']):.1f}%    | {float(r['pnl']):.2f} | {float(r['roi']):.2f}%"
        )
    return 0


def run_grid_search_multi_season(
    season_codes: List[str],
    train_ratio: float,
    markets: List[str],
    min_edges: List[float],
    odds_mins: List[float],
    odds_maxs: List[float],
    include_leagues: Set[str],
    exclude_leagues: Set[str],
    include_weekdays: Set[int],
    exclude_weekdays: Set[int],
    min_bets_per_season: int,
    top_n: int,
) -> int:
    season_rows: Dict[str, List[dict]] = {}
    for s in season_codes:
        rows = _load_rows(s)
        if not rows:
            print(f"Skipped season {s}: no rows.")
            continue
        season_rows[s] = rows
    if not season_rows:
        print("No seasons loaded for multi-season grid.")
        return 1

    ranked: List[Dict[str, object]] = []
    for min_edge, odds_min, odds_max in product(min_edges, odds_mins, odds_maxs):
        if odds_max > 0 and odds_min > 0 and odds_min > odds_max:
            continue
        per_season = []
        ok = True
        for season_code, rows in season_rows.items():
            res = evaluate_backtest(
                rows=rows,
                season_code=season_code,
                train_ratio=train_ratio,
                min_edge=min_edge,
                markets=markets,
                odds_min=odds_min,
                odds_max=odds_max,
                include_leagues=include_leagues,
                exclude_leagues=exclude_leagues,
                include_weekdays=include_weekdays,
                exclude_weekdays=exclude_weekdays,
            )
            if "error" in res:
                ok = False
                break
            if int(res["bets"]) < min_bets_per_season:
                ok = False
                break
            per_season.append(res)
        if not ok or not per_season:
            continue

        rois = [float(r["roi"]) for r in per_season]
        pnls = [float(r["pnl"]) for r in per_season]
        bets = [int(r["bets"]) for r in per_season]
        total_bets = sum(bets)
        total_pnl = sum(pnls)
        weighted_roi = (total_pnl / total_bets) * 100.0 if total_bets else 0.0
        min_roi = min(rois)
        roi_std = pstdev(rois) if len(rois) > 1 else 0.0
        positive_seasons = sum(1 for r in rois if r > 0.0)

        ranked.append(
            {
                "cfg_min_edge": min_edge,
                "cfg_odds_min": odds_min,
                "cfg_odds_max": odds_max,
                "seasons": len(per_season),
                "positive_seasons": positive_seasons,
                "total_bets": total_bets,
                "total_pnl": total_pnl,
                "weighted_roi": weighted_roi,
                "min_roi": min_roi,
                "roi_std": roi_std,
            }
        )

    if not ranked:
        print("No multi-season grid configurations met constraints.")
        return 0

    ranked.sort(
        key=lambda r: (
            float(r["min_roi"]),
            float(r["weighted_roi"]),
            -float(r["roi_std"]),
            int(r["total_bets"]),
        ),
        reverse=True,
    )
    k = min(top_n, len(ranked))
    print(f"Multi-Season Grid Top {k}")
    print("min_edge | odds_min | odds_max | seasons+ | min_roi | w_roi | roi_std | bets | pnl_u")
    print("----------------------------------------------------------------------------------------")
    for r in ranked[:k]:
        print(
            f"{r['cfg_min_edge']:.1f}     | {r['cfg_odds_min']:.2f}     | {r['cfg_odds_max']:.2f}     | "
            f"{int(r['positive_seasons'])}/{int(r['seasons'])}     | {float(r['min_roi']):.2f}%  | "
            f"{float(r['weighted_roi']):.2f}% | {float(r['roi_std']):.2f}    | "
            f"{int(r['total_bets'])}  | {float(r['total_pnl']):.2f}"
        )
    return 0


def _best_multi_season_config_for_market(
    market: str,
    season_rows: Dict[str, List[dict]],
    train_ratio: float,
    min_edges: List[float],
    odds_mins: List[float],
    odds_maxs: List[float],
    include_leagues: Set[str],
    exclude_leagues: Set[str],
    include_weekdays: Set[int],
    exclude_weekdays: Set[int],
    min_bets_per_season: int,
) -> Dict[str, object]:
    ranked: List[Dict[str, object]] = []
    for min_edge, odds_min, odds_max in product(min_edges, odds_mins, odds_maxs):
        if odds_max > 0 and odds_min > 0 and odds_min > odds_max:
            continue
        per_season = []
        ok = True
        for season_code, rows in season_rows.items():
            res = evaluate_backtest(
                rows=rows,
                season_code=season_code,
                train_ratio=train_ratio,
                min_edge=min_edge,
                markets=[market],
                odds_min=odds_min,
                odds_max=odds_max,
                include_leagues=include_leagues,
                exclude_leagues=exclude_leagues,
                include_weekdays=include_weekdays,
                exclude_weekdays=exclude_weekdays,
            )
            if "error" in res or int(res["bets"]) < min_bets_per_season:
                ok = False
                break
            per_season.append(res)
        if not ok or not per_season:
            continue
        rois = [float(r["roi"]) for r in per_season]
        pnls = [float(r["pnl"]) for r in per_season]
        bets = [int(r["bets"]) for r in per_season]
        total_bets = sum(bets)
        total_pnl = sum(pnls)
        weighted_roi = (total_pnl / total_bets) * 100.0 if total_bets else 0.0
        ranked.append(
            {
                "market": market,
                "cfg_min_edge": min_edge,
                "cfg_odds_min": odds_min,
                "cfg_odds_max": odds_max,
                "seasons": len(per_season),
                "positive_seasons": sum(1 for r in rois if r > 0.0),
                "min_roi": min(rois),
                "weighted_roi": weighted_roi,
                "roi_std": pstdev(rois) if len(rois) > 1 else 0.0,
                "total_bets": total_bets,
                "total_pnl": total_pnl,
            }
        )

    if not ranked:
        return {
            "market": market,
            "cfg_min_edge": 0.0,
            "cfg_odds_min": 0.0,
            "cfg_odds_max": 0.0,
            "seasons": len(season_rows),
            "positive_seasons": 0,
            "min_roi": -999.0,
            "weighted_roi": -999.0,
            "roi_std": 0.0,
            "total_bets": 0,
            "total_pnl": 0.0,
        }

    ranked.sort(
        key=lambda r: (
            float(r["min_roi"]),
            float(r["weighted_roi"]),
            -float(r["roi_std"]),
            int(r["total_bets"]),
        ),
        reverse=True,
    )
    return ranked[0]


def run_market_leaderboard_multi_season(
    season_codes: List[str],
    train_ratio: float,
    min_edges: List[float],
    odds_mins: List[float],
    odds_maxs: List[float],
    include_leagues: Set[str],
    exclude_leagues: Set[str],
    include_weekdays: Set[int],
    exclude_weekdays: Set[int],
    min_bets_per_season: int,
    top_n: int,
) -> int:
    season_rows: Dict[str, List[dict]] = {}
    for s in season_codes:
        rows = _load_rows(s)
        if rows:
            season_rows[s] = rows
    if not season_rows:
        print("No seasons loaded for market leaderboard.")
        return 1

    best_by_market: List[Dict[str, object]] = []
    for m in MARKET_KEYS:
        best = _best_multi_season_config_for_market(
            market=m,
            season_rows=season_rows,
            train_ratio=train_ratio,
            min_edges=min_edges,
            odds_mins=odds_mins,
            odds_maxs=odds_maxs,
            include_leagues=include_leagues,
            exclude_leagues=exclude_leagues,
            include_weekdays=include_weekdays,
            exclude_weekdays=exclude_weekdays,
            min_bets_per_season=min_bets_per_season,
        )
        best_by_market.append(best)

    best_by_market.sort(
        key=lambda r: (
            float(r["min_roi"]),
            float(r["weighted_roi"]),
            int(r["positive_seasons"]),
        ),
        reverse=True,
    )
    k = min(top_n, len(best_by_market))
    print(f"Per-Market Multi-Season Leaderboard (Top {k})")
    print("market | seasons+ | min_roi | w_roi | bets | pnl_u | min_edge | odds_min | odds_max")
    print("---------------------------------------------------------------------------------------")
    for r in best_by_market[:k]:
        print(
            f"{r['market']:<10} | {int(r['positive_seasons'])}/{int(r['seasons'])}     | "
            f"{float(r['min_roi']):.2f}%  | {float(r['weighted_roi']):.2f}% | "
            f"{int(r['total_bets'])}  | {float(r['total_pnl']):.2f} | "
            f"{float(r['cfg_min_edge']):.1f}     | {float(r['cfg_odds_min']):.2f}     | {float(r['cfg_odds_max']):.2f}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest soccer bot with historical odds.")
    p.add_argument("--season-code", default="2526", help="football-data season code")
    p.add_argument("--train-ratio", type=float, default=0.70, help="fraction for train split")
    p.add_argument("--min-edge", type=float, default=3.0, help="minimum edge percentage")
    p.add_argument(
        "--markets",
        default="all",
        help="comma list: all,home,dnb,o15,o25,o35,u25,u35,btts_yes,btts_no,co85,co95,co105,cu95,cu105",
    )
    p.add_argument("--odds-min", type=float, default=0.0, help="minimum bookmaker odds filter")
    p.add_argument("--odds-max", type=float, default=0.0, help="maximum bookmaker odds filter (0 disables)")
    p.add_argument("--include-leagues", default="", help="comma-separated leagues to include")
    p.add_argument("--exclude-leagues", default="", help="comma-separated leagues to exclude")
    p.add_argument("--include-weekdays", default="", help="comma weekdays include (mon,tue,... or 0-6)")
    p.add_argument("--exclude-weekdays", default="", help="comma weekdays exclude (mon,tue,... or 0-6)")
    p.add_argument("--grid-search", action="store_true", help="run parameter grid search")
    p.add_argument("--grid-min-edges", default="3,5,7,9", help="comma list for grid min-edge")
    p.add_argument("--grid-odds-min", default="0,1.6,1.8,2.0", help="comma list for grid odds-min")
    p.add_argument("--grid-odds-max", default="0,2.2,2.6,3.2", help="comma list for grid odds-max")
    p.add_argument("--grid-min-bets", type=int, default=80, help="min bets threshold for grid results")
    p.add_argument("--grid-top", type=int, default=10, help="top N grid rows to print")
    p.add_argument(
        "--season-codes",
        default="",
        help="comma list of seasons for multi-season grid (example: 2324,2425,2526)",
    )
    p.add_argument(
        "--grid-min-bets-per-season",
        type=int,
        default=60,
        help="for multi-season grid: minimum bets required in each season.",
    )
    p.add_argument(
        "--market-leaderboard",
        action="store_true",
        help="for multi-season mode: find best robust config per individual market.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mkts = expand_markets([m.strip().lower() for m in args.markets.split(",") if m.strip()])
    include_leagues = _parse_csv_set(args.include_leagues)
    exclude_leagues = _parse_csv_set(args.exclude_leagues)
    include_weekdays = _parse_weekday_set(args.include_weekdays)
    exclude_weekdays = _parse_weekday_set(args.exclude_weekdays)

    if args.grid_search:
        min_edges = [_to_float(x, 0.0) for x in args.grid_min_edges.split(",") if x.strip()]
        odds_mins = [_to_float(x, 0.0) for x in args.grid_odds_min.split(",") if x.strip()]
        odds_maxs = [_to_float(x, 0.0) for x in args.grid_odds_max.split(",") if x.strip()]
        season_codes = [s.strip() for s in args.season_codes.split(",") if s.strip()]
        if season_codes:
            if args.market_leaderboard:
                raise SystemExit(
                    run_market_leaderboard_multi_season(
                        season_codes=season_codes,
                        train_ratio=args.train_ratio,
                        min_edges=min_edges or [args.min_edge],
                        odds_mins=odds_mins or [args.odds_min],
                        odds_maxs=odds_maxs or [args.odds_max],
                        include_leagues=include_leagues,
                        exclude_leagues=exclude_leagues,
                        include_weekdays=include_weekdays,
                        exclude_weekdays=exclude_weekdays,
                        min_bets_per_season=max(0, args.grid_min_bets_per_season),
                        top_n=max(1, args.grid_top),
                    )
                )
            raise SystemExit(
                run_grid_search_multi_season(
                    season_codes=season_codes,
                    train_ratio=args.train_ratio,
                    markets=mkts,
                    min_edges=min_edges or [args.min_edge],
                    odds_mins=odds_mins or [args.odds_min],
                    odds_maxs=odds_maxs or [args.odds_max],
                    include_leagues=include_leagues,
                    exclude_leagues=exclude_leagues,
                    include_weekdays=include_weekdays,
                    exclude_weekdays=exclude_weekdays,
                    min_bets_per_season=max(0, args.grid_min_bets_per_season),
                    top_n=max(1, args.grid_top),
                )
            )
        rows = _load_rows(args.season_code)
        if not rows:
            print("No historical rows downloaded.")
            raise SystemExit(1)
        raise SystemExit(
            run_grid_search(
                rows=rows,
                season_code=args.season_code,
                train_ratio=args.train_ratio,
                markets=mkts,
                min_edges=min_edges or [args.min_edge],
                odds_mins=odds_mins or [args.odds_min],
                odds_maxs=odds_maxs or [args.odds_max],
                include_leagues=include_leagues,
                exclude_leagues=exclude_leagues,
                include_weekdays=include_weekdays,
                exclude_weekdays=exclude_weekdays,
                min_bets=max(0, args.grid_min_bets),
                top_n=max(1, args.grid_top),
            )
        )

    rows = _load_rows(args.season_code)
    if not rows:
        print("No historical rows downloaded.")
        raise SystemExit(1)
    result = evaluate_backtest(
        rows=rows,
        season_code=args.season_code,
        train_ratio=args.train_ratio,
        min_edge=args.min_edge,
        markets=mkts,
        odds_min=args.odds_min,
        odds_max=args.odds_max,
        include_leagues=include_leagues,
        exclude_leagues=exclude_leagues,
        include_weekdays=include_weekdays,
        exclude_weekdays=exclude_weekdays,
    )
    raise SystemExit(print_backtest_result(result))
