from __future__ import annotations

import csv
import os
import random
import re
import subprocess
import sys
from datetime import date, timedelta
from typing import Dict, List, Tuple

import streamlit as st
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ODDS_CSV = "upcoming_odds.csv"
BETS_CSV = "bets_log.csv"
MARKET_CHOICES: List[Tuple[str, str]] = [
    ("home", "Home (Winner)"),
    ("away", "Away (Winner)"),
    ("dnb", "Home (DNB)"),
    ("adnb", "Away (DNB)"),
    ("ah_h_m0_5", "AH Home -0.5"),
    ("ah_a_p0_5", "AH Away +0.5"),
    ("ah_h_m1_5", "AH Home -1.5"),
    ("ah_a_p1_5", "AH Away +1.5"),
    ("ahc_h_m1_5", "AH Corners Home -1.5"),
    ("ahc_a_p1_5", "AH Corners Away +1.5"),
    ("ahc_h_m2_5", "AH Corners Home -2.5"),
    ("ahc_a_p2_5", "AH Corners Away +2.5"),
    ("o15", "Over 1.5 Goals"),
    ("o25", "Over 2.5 Goals"),
    ("o35", "Over 3.5 Goals"),
    ("u25", "Under 2.5 Goals"),
    ("u35", "Under 3.5 Goals"),
    ("btts_yes", "BTTS Yes"),
    ("btts_no", "BTTS No"),
    ("co85", "Over 8.5 Corners"),
    ("co95", "Over 9.5 Corners"),
    ("co105", "Over 10.5 Corners"),
    ("cu95", "Under 9.5 Corners"),
    ("cu105", "Under 10.5 Corners"),
]
BACKTEST_MARKETS = [
    "home",
    "away",
    "dnb",
    "adnb",
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
WEEKDAY_CHOICES: List[Tuple[str, str]] = [
    ("mon", "Monday"),
    ("tue", "Tuesday"),
    ("wed", "Wednesday"),
    ("thu", "Thursday"),
    ("fri", "Friday"),
    ("sat", "Saturday"),
    ("sun", "Sunday"),
]


def configured_app_passcode() -> str:
    try:
        val = (st.secrets.get("APP_PASSCODE", "") or "").strip()
        if val:
            return val
    except Exception:
        pass
    return (os.getenv("APP_PASSCODE", "") or "").strip()


def require_access_gate() -> bool:
    expected = configured_app_passcode()
    if not expected:
        return True
    if st.session_state.get("access_granted"):
        return True

    st.title("Soccer Bot Free")
    st.subheader("Access Required")
    entered = st.text_input("Passcode", type="password", key="access_passcode_input")
    if st.button("Unlock", type="primary", use_container_width=True):
        if entered == expected:
            st.session_state.access_granted = True
            st.success("Access granted.")
            st.rerun()
        else:
            st.error("Invalid passcode.")
    st.info("Ask the app owner for the passcode.")
    return False


def to_float(raw: str, default: float = 0.0) -> float:
    try:
        return float((raw or "").strip())
    except (TypeError, ValueError):
        return default


def date_from_choice(choice: str, custom_val: date, allow_all: bool = False) -> str:
    c = (choice or "").strip().lower()
    if allow_all and c == "all":
        return "all"
    if c == "today":
        return "today"
    if c == "tomorrow":
        return "tomorrow"
    if c == "yesterday":
        return "yesterday"
    return custom_val.isoformat()


def run_cmd(args: List[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [PY] + args,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def ensure_bootstrap_files() -> None:
    odds_path = os.path.join(BASE_DIR, ODDS_CSV)
    if not os.path.exists(odds_path):
        example = os.path.join(BASE_DIR, "upcoming_odds.example.csv")
        if os.path.exists(example):
            with open(example, "r", encoding="utf-8") as src, open(odds_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())


def odds_fixture_count() -> int:
    path = os.path.join(BASE_DIR, ODDS_CSV)
    if not os.path.exists(path):
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def bets_stats() -> Dict[str, int]:
    stats = {"total": 0, "settled": 0, "pending": 0}
    path = os.path.join(BASE_DIR, BETS_CSV)
    if not os.path.exists(path):
        return stats
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats["total"] += 1
            settled = (row.get("settled") or "").strip().lower() in {"1", "true", "yes", "y"}
            if settled:
                stats["settled"] += 1
            else:
                stats["pending"] += 1
    return stats


def load_played_picks(date_filter: str = "all", profile_filter: str = "") -> List[Dict[str, str]]:
    path = os.path.join(BASE_DIR, BETS_CSV)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            settled = (row.get("settled") or "").strip().lower() in {"1", "true", "yes", "y"}
            if not settled:
                continue
            if date_filter != "all" and (row.get("date") or "").strip() != date_filter.strip():
                continue
            if profile_filter and (row.get("profile") or "").strip() != profile_filter.strip():
                continue
            result = (row.get("result") or "").strip().lower()
            if result not in {"win", "loss", "lost", "won", "push"}:
                pnl = to_float(row.get("pnl_units", "0"))
                if pnl > 0:
                    result = "win"
                elif pnl < 0:
                    result = "loss"
                else:
                    result = "push"
            out.append(
                {
                    "Date": (row.get("date") or "").strip(),
                    "League": (row.get("league") or "").strip(),
                    "Fixture": (row.get("fixture") or "").strip(),
                    "Market": (row.get("market") or "").strip(),
                    "Result": result.upper(),
                    "Book": f"{to_float(row.get('book_odds', '0')):.2f}",
                    "StakeU": f"{to_float(row.get('stake_units', '0')):.2f}",
                    "PnL U": f"{to_float(row.get('pnl_units', '0')):+.2f}",
                    "Edge": f"{to_float(row.get('edge_pct', '0')):.1f}%",
                    "Confidence": f"{to_float(row.get('confidence_pct', '0')):.1f}%",
                    "Profile": (row.get("profile") or "").strip(),
                }
            )
    return out


def load_parlay_results(date_filter: str = "all", profile_filter: str = "") -> List[Dict[str, str]]:
    path = os.path.join(BASE_DIR, BETS_CSV)
    if not os.path.exists(path):
        return []

    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            logged_at = (row.get("logged_at") or "").strip()
            bet_date = (row.get("date") or "").strip()
            profile = (row.get("profile") or "").strip()
            if not logged_at or not bet_date:
                continue
            if date_filter != "all" and bet_date != date_filter.strip():
                continue
            if profile_filter and profile != profile_filter.strip():
                continue
            key = (logged_at, bet_date, profile)
            groups.setdefault(key, []).append(row)

    out: List[Dict[str, str]] = []
    for (logged_at, bet_date, profile), legs in groups.items():
        # Parlay ticket definition: 2+ picks logged in one run.
        if len(legs) < 2:
            continue
        wins = 0
        losses = 0
        pushes = 0
        pending = 0
        combined_odds = 1.0
        for r in legs:
            settled = (r.get("settled") or "").strip().lower() in {"1", "true", "yes", "y"}
            if not settled:
                pending += 1
                continue
            result = (r.get("result") or "").strip().lower()
            if result in {"win", "won", "w"}:
                wins += 1
                combined_odds *= max(1.0, to_float(r.get("book_odds", "1")))
            elif result in {"loss", "lost", "l"}:
                losses += 1
            else:
                # push/void leg -> multiplier 1.0
                pushes += 1

        if losses > 0:
            status = "LOST"
            pnl_u = -1.0
        elif pending > 0:
            status = "PENDING"
            pnl_u = 0.0
        elif wins > 0:
            status = "WON"
            pnl_u = combined_odds - 1.0
        else:
            status = "PUSH"
            pnl_u = 0.0

        out.append(
            {
                "Date": bet_date,
                "Ticket ID": f"{logged_at} | {profile or 'manual'}",
                "Logged At": logged_at,
                "Profile": profile or "manual",
                "Legs": str(len(legs)),
                "Wins": str(wins),
                "Losses": str(losses),
                "Pushes": str(pushes),
                "Pending": str(pending),
                "Fixtures": ", ".join(sorted({(x.get("fixture") or "").strip() for x in legs})[:3])
                + (" ..." if len({(x.get("fixture") or "").strip() for x in legs}) > 3 else ""),
                "Parlay Odds": f"{combined_odds:.2f}" if status == "WON" else "-",
                "Parlay Result": status,
                "Parlay PnL U (1u)": f"{pnl_u:+.2f}",
            }
        )
    return out


def load_parlay_leg_rows(date_filter: str = "all", profile_filter: str = "") -> List[Dict[str, str]]:
    path = os.path.join(BASE_DIR, BETS_CSV)
    if not os.path.exists(path):
        return []

    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            logged_at = (row.get("logged_at") or "").strip()
            bet_date = (row.get("date") or "").strip()
            profile = (row.get("profile") or "").strip()
            if not logged_at or not bet_date:
                continue
            if date_filter != "all" and bet_date != date_filter.strip():
                continue
            if profile_filter and profile != profile_filter.strip():
                continue
            groups.setdefault((logged_at, bet_date, profile), []).append(row)

    legs_out: List[Dict[str, str]] = []
    for (logged_at, bet_date, profile), legs in groups.items():
        if len(legs) < 2:
            continue
        ticket_id = f"{logged_at} | {profile or 'manual'}"
        for r in legs:
            settled = (r.get("settled") or "").strip().lower() in {"1", "true", "yes", "y"}
            result = (r.get("result") or "").strip().upper() if settled else "PENDING"
            if not result:
                pnl = to_float(r.get("pnl_units", "0"))
                result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "PUSH")
            legs_out.append(
                {
                    "Ticket ID": ticket_id,
                    "Date": bet_date,
                    "League": (r.get("league") or "").strip(),
                    "Fixture": (r.get("fixture") or "").strip(),
                    "Market": (r.get("market") or "").strip(),
                    "Book": f"{to_float(r.get('book_odds', '0')):.2f}",
                    "Result": result,
                    "Edge": f"{to_float(r.get('edge_pct', '0')):.1f}%",
                    "Confidence": f"{to_float(r.get('confidence_pct', '0')):.1f}%",
                }
            )
    return legs_out


def parse_scan_summary(output: str) -> Tuple[int, int]:
    # Returns (all_qualifying_estimate, shortlist_count)
    shortlist = 0
    m = re.search(r"PARLAY SHORTLIST \(TOP (\d+)\)", output)
    if m:
        shortlist = int(m.group(1))
    total = 0
    if "SNIPER REPORT (VALUE FOUND)" in output:
        report = output.split("SNIPER REPORT (VALUE FOUND)", 1)[1]
        if "PARLAY SHORTLIST" in report:
            report = report.split("PARLAY SHORTLIST", 1)[0]
        lines = [ln for ln in report.splitlines() if " | " in ln and re.match(r"^\d{4}-\d{2}-\d{2}\s+\|", ln)]
        total = len(lines)
    return total, shortlist


def _parse_pct(raw: str) -> float:
    try:
        return float((raw or "").replace("%", "").replace("+", "").strip())
    except (TypeError, ValueError):
        return 0.0


def parse_grid_rows(output: str, min_bets: int) -> List[Dict[str, str]]:
    lines = output.splitlines()
    header_idx = -1
    headers: List[str] = []
    for i, ln in enumerate(lines):
        if "|" in ln and "min_edge" in ln and "odds_min" in ln and "odds_max" in ln:
            header_idx = i
            headers = [h.strip() for h in ln.split("|")]
            break
    if header_idx < 0:
        return []

    rows: List[Dict[str, str]] = []
    for ln in lines[header_idx + 1 :]:
        if not ln.strip():
            continue
        if set(ln.strip()) == {"-"}:
            continue
        if "|" not in ln:
            # Stop once we leave the table body.
            if rows:
                break
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != len(headers):
            continue
        row = dict(zip(headers, parts))
        bets = int(float(row.get("bets", "0") or 0))
        roi_key = "w_roi" if "w_roi" in row else ("roi" if "roi" in row else "")
        min_roi_key = "min_roi" if "min_roi" in row else ""
        roi_val = _parse_pct(row.get(roi_key, "0")) if roi_key else 0.0
        min_roi_val = _parse_pct(row.get(min_roi_key, "0")) if min_roi_key else roi_val
        candidate = (bets >= min_bets) and (roi_val > 0.0) and (min_roi_val > 0.0)
        row["deploy_candidate"] = "YES" if candidate else "NO"
        rows.append(row)
    return rows


def parse_single_backtest_metrics(output: str) -> Dict[str, float]:
    # Parses: Bets: 98 | Wins: 49 | Hit rate: 50.0% | PnL (1u flat): -3.89u | ROI: -3.97%
    m = re.search(
        r"Bets:\s*(\d+)\s*\|\s*Wins:\s*(\d+)\s*\|\s*Hit rate:\s*([-\d.]+)%\s*\|\s*PnL.*?:\s*([-\d.]+)u\s*\|\s*ROI:\s*([-\d.]+)%",
        output,
    )
    if not m:
        return {"bets": 0.0, "wins": 0.0, "hit_rate": 0.0, "pnl": 0.0, "roi": -999.0}
    return {
        "bets": float(m.group(1)),
        "wins": float(m.group(2)),
        "hit_rate": float(m.group(3)),
        "pnl": float(m.group(4)),
        "roi": float(m.group(5)),
    }


def parse_report_table(output: str, marker: str) -> List[Dict[str, str]]:
    lines = output.splitlines()
    marker_idx = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith(marker):
            marker_idx = i
            break
    if marker_idx < 0:
        return []

    header_idx = -1
    for i in range(marker_idx + 1, len(lines)):
        if " | " in lines[i] and "Date" in lines[i] and "Market" in lines[i]:
            header_idx = i
            break
    if header_idx < 0:
        return []
    headers = [h.strip() for h in lines[header_idx].split(" | ")]

    rows: List[Dict[str, str]] = []
    for i in range(header_idx + 1, len(lines)):
        ln = lines[i]
        if not ln.strip():
            break
        if ln.startswith("PARLAY SHORTLIST") or ln.startswith("Corner Predictions") or ln.startswith("SNIPER REPORT"):
            break
        if set(ln.strip()) == {"-"}:
            continue
        if " | " not in ln:
            continue
        parts = [p.strip() for p in ln.split(" | ")]
        if len(parts) != len(headers):
            continue
        rows.append(dict(zip(headers, parts)))
    return rows


def rows_to_df(rows: List[Dict[str, str]], sort_col: str = "Edge"):
    if not rows:
        return None
    if pd is None:
        return rows
    df = pd.DataFrame(rows)
    if sort_col in df.columns:
        cleaned = (
            df[sort_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df["_sort"] = pd.to_numeric(cleaned, errors="coerce")
        df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return df


def render_terminal_output(title: str, cmd: List[str], output: str, rc: int) -> None:
    with st.expander(title, expanded=True):
        st.code(f"$ python {' '.join(cmd)}\n{output}", language="bash")
    if rc == 0:
        st.success("Completed")
    else:
        st.error("Failed")


def build_scan_cmd(cfg: Dict[str, object]) -> List[str]:
    cmd: List[str] = [
        "main.py",
        "--color",
        "never",
        "--season-code",
        str(cfg["season_code"]),
        "--odds-csv",
        ODDS_CSV,
        "--date",
        str(cfg["scan_date"]),
        "--mode",
        str(cfg["mode"]),
        "--min-edge",
        str(cfg["min_edge"]),
        "--max-picks",
        str(cfg["max_picks"]),
        "--shortlist-market-cap",
        str(cfg["shortlist_market_cap"]),
        "--bankroll",
        str(cfg["bankroll"]),
    ]
    if cfg["one_pick_per_match"]:
        cmd.append("--one-pick-per-match")
    if cfg["log_picks"]:
        cmd.append("--log-picks")
    if cfg["odds_min"] > 0:
        cmd += ["--odds-min", str(cfg["odds_min"])]
    if cfg["odds_max"] > 0:
        cmd += ["--odds-max", str(cfg["odds_max"])]
    if cfg["include_markets"].strip():
        cmd += ["--include-markets", str(cfg["include_markets"]).strip()]
    if cfg["exclude_markets"].strip():
        cmd += ["--exclude-markets", str(cfg["exclude_markets"]).strip()]
    if cfg["profile"] != "none":
        cmd += ["--profile", str(cfg["profile"])]
    if cfg["min_confidence"] > 0:
        cmd += ["--min-confidence", str(cfg["min_confidence"])]
    if cfg["max_confidence"] > 0:
        cmd += ["--max-confidence", str(cfg["max_confidence"])]
    if str(cfg["scan_include_weekdays"]).strip():
        cmd += ["--include-weekdays", str(cfg["scan_include_weekdays"]).strip()]
    if str(cfg["scan_exclude_weekdays"]).strip():
        cmd += ["--exclude-weekdays", str(cfg["scan_exclude_weekdays"]).strip()]
    return cmd


def app_css() -> None:
    st.markdown(
        """
<style>
.card {
    border: 1px solid #e3e7ef;
    border-radius: 12px;
    padding: 14px 16px;
    background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
}
.label { color: #51606f; font-size: 0.85rem; margin-bottom: 4px; }
.value { color: #0f172a; font-size: 1.25rem; font-weight: 700; }
</style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Soccer Bot Free", page_icon="⚽", layout="wide")
if not require_access_gate():
    st.stop()
app_css()
ensure_bootstrap_files()

if "preset" not in st.session_state:
    st.session_state.preset = "Balanced"
if "mode" not in st.session_state:
    st.session_state.mode = "aggressive"
if "min_edge" not in st.session_state:
    st.session_state.min_edge = 8.0
if "odds_min" not in st.session_state:
    st.session_state.odds_min = 0.0
if "odds_max" not in st.session_state:
    st.session_state.odds_max = 0.0
if "include_market_keys" not in st.session_state:
    st.session_state.include_market_keys = []
if "exclude_market_keys" not in st.session_state:
    st.session_state.exclude_market_keys = []
if "refresh_lookahead_days" not in st.session_state:
    st.session_state.refresh_lookahead_days = 7
if "auto_fill_margin" not in st.session_state:
    st.session_state.auto_fill_margin = -8.0
if "one_pick_per_match" not in st.session_state:
    st.session_state.one_pick_per_match = True
if "shortlist_market_cap" not in st.session_state:
    st.session_state.shortlist_market_cap = 3
if "max_picks" not in st.session_state:
    st.session_state.max_picks = 10
if "scan_date" not in st.session_state:
    st.session_state.scan_date = "today"
if "log_picks" not in st.session_state:
    st.session_state.log_picks = True
if "scan_date_mode" not in st.session_state:
    st.session_state.scan_date_mode = "Today"
if "scan_date_custom" not in st.session_state:
    st.session_state.scan_date_custom = date.today()
if "settle_date_mode" not in st.session_state:
    st.session_state.settle_date_mode = "Yesterday"
if "settle_date_custom" not in st.session_state:
    st.session_state.settle_date_custom = date.today() - timedelta(days=1)
if "report_date_mode" not in st.session_state:
    st.session_state.report_date_mode = "All"
if "report_date_custom" not in st.session_state:
    st.session_state.report_date_custom = date.today()
if "played_date_mode" not in st.session_state:
    st.session_state.played_date_mode = "All"
if "played_date_custom" not in st.session_state:
    st.session_state.played_date_custom = date.today()
if "last_scan_output" not in st.session_state:
    st.session_state.last_scan_output = ""
if "last_scan_cmd" not in st.session_state:
    st.session_state.last_scan_cmd = ""
if "min_confidence" not in st.session_state:
    st.session_state.min_confidence = 0.0
if "max_confidence" not in st.session_state:
    st.session_state.max_confidence = 0.0
if "scan_include_weekdays" not in st.session_state:
    st.session_state.scan_include_weekdays = []
if "scan_exclude_weekdays" not in st.session_state:
    st.session_state.scan_exclude_weekdays = []
if "bt_markets" not in st.session_state:
    st.session_state.bt_markets = ["o25"]
if "bt_train_ratio" not in st.session_state:
    st.session_state.bt_train_ratio = 0.70
if "bt_min_edge" not in st.session_state:
    st.session_state.bt_min_edge = 6.0
if "bt_odds_min" not in st.session_state:
    st.session_state.bt_odds_min = 0.0
if "bt_odds_max" not in st.session_state:
    st.session_state.bt_odds_max = 0.0
if "bt_season_codes" not in st.session_state:
    st.session_state.bt_season_codes = "2324,2425,2526"
if "bt_include_weekdays" not in st.session_state:
    st.session_state.bt_include_weekdays = ""
if "bt_exclude_weekdays" not in st.session_state:
    st.session_state.bt_exclude_weekdays = ""
if "bt_include_leagues" not in st.session_state:
    st.session_state.bt_include_leagues = ""
if "bt_exclude_leagues" not in st.session_state:
    st.session_state.bt_exclude_leagues = ""
if "bt_grid_min_edges" not in st.session_state:
    st.session_state.bt_grid_min_edges = "3,5,7,9"
if "bt_grid_odds_min" not in st.session_state:
    st.session_state.bt_grid_odds_min = "1.6,1.8,2.0"
if "bt_grid_odds_max" not in st.session_state:
    st.session_state.bt_grid_odds_max = "2.2,2.6,3.2"
if "bt_grid_min_bets" not in st.session_state:
    st.session_state.bt_grid_min_bets = 60
if "bt_grid_top" not in st.session_state:
    st.session_state.bt_grid_top = 10
if "bt_auto_tries" not in st.session_state:
    st.session_state.bt_auto_tries = 40
if "bt_auto_min_bets" not in st.session_state:
    st.session_state.bt_auto_min_bets = 60

st.title("Soccer Bot Free")
st.caption("Scanner + settlement control panel")

with st.sidebar:
    if configured_app_passcode() and st.session_state.get("access_granted"):
        if st.button("Lock App", use_container_width=True):
            st.session_state.access_granted = False
            st.rerun()
    st.subheader("Scan Settings")
    preset = st.selectbox(
        "Preset",
        ["DailyScan Morning (11 Feb style)", "O25 Robust 2425-2526", "Balanced", "Conservative", "Custom"],
        key="preset",
    )
    if st.button("Apply Preset", use_container_width=True):
        if preset == "DailyScan Morning (11 Feb style)":
            st.session_state.mode = "aggressive"
            st.session_state.min_edge = 8.0
            st.session_state.odds_min = 0.0
            st.session_state.odds_max = 0.0
            st.session_state.include_market_keys = []
            st.session_state.exclude_market_keys = []
            st.session_state.refresh_lookahead_days = 7
            st.session_state.auto_fill_margin = -8.0
            st.session_state.one_pick_per_match = True
            st.session_state.shortlist_market_cap = 3
            st.session_state.max_picks = 10
            st.session_state.scan_date_mode = "Today"
            st.session_state.scan_date_custom = date.today()
            st.session_state.log_picks = True
            st.session_state.min_confidence = 0.0
            st.session_state.max_confidence = 0.0
            st.session_state.scan_include_weekdays = []
            st.session_state.scan_exclude_weekdays = []
        elif preset == "O25 Robust 2425-2526":
            st.session_state.mode = "balanced"
            st.session_state.min_edge = 3.0
            st.session_state.odds_min = 2.0
            st.session_state.odds_max = 2.6
            st.session_state.include_market_keys = ["o25"]
            st.session_state.exclude_market_keys = []
            st.session_state.refresh_lookahead_days = 7
            st.session_state.auto_fill_margin = -8.0
            st.session_state.one_pick_per_match = True
            st.session_state.shortlist_market_cap = 1
            st.session_state.max_picks = 8
            st.session_state.scan_date_mode = "Today"
            st.session_state.scan_date_custom = date.today()
            st.session_state.log_picks = True
            st.session_state.min_confidence = 0.0
            st.session_state.max_confidence = 0.0
            st.session_state.scan_include_weekdays = []
            st.session_state.scan_exclude_weekdays = ["tue", "wed", "thu"]
        elif preset == "Balanced":
            st.session_state.mode = "balanced"
            st.session_state.min_edge = 6.0
            st.session_state.odds_min = 1.8
            st.session_state.odds_max = 2.8
            st.session_state.include_market_keys = ["o25", "dnb", "home", "away", "btts_yes", "btts_no"]
            st.session_state.exclude_market_keys = ["co105", "ahc_h_m2_5", "ahc_a_p2_5"]
            st.session_state.refresh_lookahead_days = 7
            st.session_state.auto_fill_margin = -8.0
            st.session_state.one_pick_per_match = True
            st.session_state.shortlist_market_cap = 3
            st.session_state.max_picks = 10
            st.session_state.scan_date_mode = "Today"
            st.session_state.scan_date_custom = date.today()
            st.session_state.log_picks = True
            st.session_state.min_confidence = 50.0
            st.session_state.max_confidence = 0.0
            st.session_state.scan_include_weekdays = []
            st.session_state.scan_exclude_weekdays = []
        elif preset == "Conservative":
            st.session_state.mode = "safe"
            st.session_state.min_edge = 8.0
            st.session_state.odds_min = 1.6
            st.session_state.odds_max = 2.4
            st.session_state.include_market_keys = ["o15", "o25", "dnb", "adnb", "u35", "cu95", "cu105"]
            st.session_state.exclude_market_keys = ["o35", "co105", "ahc_h_m2_5", "ahc_a_p2_5", "home", "away"]
            st.session_state.refresh_lookahead_days = 7
            st.session_state.auto_fill_margin = -8.0
            st.session_state.one_pick_per_match = True
            st.session_state.shortlist_market_cap = 3
            st.session_state.max_picks = 10
            st.session_state.scan_date_mode = "Today"
            st.session_state.scan_date_custom = date.today()
            st.session_state.log_picks = True
            st.session_state.min_confidence = 55.0
            st.session_state.max_confidence = 0.0
            st.session_state.scan_include_weekdays = []
            st.session_state.scan_exclude_weekdays = []
        st.rerun()
    season_code = st.text_input("Season Code", value="2526")
    scan_date_mode = st.selectbox("Scan Date", ["Today", "Tomorrow", "Custom date"], key="scan_date_mode")
    scan_date_custom = st.date_input("Scan Date (custom)", key="scan_date_custom", disabled=scan_date_mode != "Custom date")
    scan_date = date_from_choice(scan_date_mode, scan_date_custom)
    mode = st.selectbox("Mode", ["safe", "balanced", "aggressive"], key="mode")
    min_edge = st.number_input("Min Edge %", step=0.1, key="min_edge")
    max_picks = st.number_input("Max Picks", min_value=0, step=1, key="max_picks")
    one_pick_per_match = st.checkbox("One Pick Per Match", key="one_pick_per_match")
    shortlist_market_cap = st.number_input("Shortlist Market Cap", min_value=0, step=1, key="shortlist_market_cap")
    log_picks = st.checkbox("Log Picks", key="log_picks")
    bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0, step=10.0)
    profile = st.selectbox("Profile", ["none", "pro_live"], index=0)

    with st.expander("Advanced Filters", expanded=False):
        odds_min = st.number_input("Odds Min", min_value=0.0, step=0.1, key="odds_min")
        odds_max = st.number_input("Odds Max", min_value=0.0, step=0.1, key="odds_max")
        include_market_keys = st.multiselect(
            "Include Markets (optional)",
            options=[k for k, _ in MARKET_CHOICES],
            default=st.session_state.include_market_keys,
            format_func=lambda x: f"{x} - {dict(MARKET_CHOICES).get(x, x)}",
            key="include_market_keys",
        )
        exclude_market_keys = st.multiselect(
            "Exclude Markets (optional)",
            options=[k for k, _ in MARKET_CHOICES],
            default=st.session_state.exclude_market_keys,
            format_func=lambda x: f"{x} - {dict(MARKET_CHOICES).get(x, x)}",
            key="exclude_market_keys",
        )
        refresh_lookahead_days = st.number_input("Refresh Lookahead Days", min_value=0, step=1, key="refresh_lookahead_days")
        auto_fill_margin = st.number_input("Auto-fill Margin %", step=0.5, key="auto_fill_margin")
        min_confidence = st.number_input("Min Confidence %", min_value=0.0, max_value=100.0, step=1.0, key="min_confidence")
        max_confidence = st.number_input(
            "Max Confidence %",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="max_confidence",
            help="0 disables upper cap",
        )
        scan_include_weekdays = st.multiselect(
            "Include Weekdays (optional)",
            options=[k for k, _ in WEEKDAY_CHOICES],
            default=st.session_state.scan_include_weekdays,
            format_func=lambda x: f"{x} - {dict(WEEKDAY_CHOICES).get(x, x)}",
            key="scan_include_weekdays",
        )
        scan_exclude_weekdays = st.multiselect(
            "Exclude Weekdays (optional)",
            options=[k for k, _ in WEEKDAY_CHOICES],
            default=st.session_state.scan_exclude_weekdays,
            format_func=lambda x: f"{x} - {dict(WEEKDAY_CHOICES).get(x, x)}",
            key="scan_exclude_weekdays",
        )

stats = bets_stats()
odds_rows = odds_fixture_count()
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='card'><div class='label'>Fixtures In Odds CSV</div><div class='value'>{odds_rows}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div class='label'>Logged Bets</div><div class='value'>{stats['total']}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='label'>Settled</div><div class='value'>{stats['settled']}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><div class='label'>Pending</div><div class='value'>{stats['pending']}</div></div>", unsafe_allow_html=True)

scan_cfg = {
    "season_code": season_code,
    "scan_date": scan_date,
    "mode": mode,
    "min_edge": min_edge,
    "max_picks": max_picks,
    "shortlist_market_cap": shortlist_market_cap,
    "one_pick_per_match": one_pick_per_match,
    "log_picks": log_picks,
    "odds_min": odds_min,
    "odds_max": odds_max,
    "include_markets": ",".join(include_market_keys),
    "exclude_markets": ",".join(exclude_market_keys),
    "profile": profile,
    "bankroll": bankroll,
    "min_confidence": min_confidence,
    "max_confidence": max_confidence,
    "scan_include_weekdays": ",".join(scan_include_weekdays),
    "scan_exclude_weekdays": ",".join(scan_exclude_weekdays),
}

tab_scan, tab_results, tab_backtest, tab_logs = st.tabs(["Scan", "Results", "Backtest", "Files"])

with tab_scan:
    st.subheader("Run Scanner")
    a, b = st.columns(2)
    with a:
        if st.button("Run Daily Pipeline", use_container_width=True, type="primary"):
            cmd_refresh = [
                "main.py",
                "--color",
                "never",
                "--season-code",
                season_code,
                "--odds-csv",
                ODDS_CSV,
                "--refresh-odds",
                "--refresh-date",
                scan_date,
                "--refresh-lookahead-days",
                str(refresh_lookahead_days),
                "--date",
                scan_date,
            ]
            if scan_cfg["scan_include_weekdays"].strip():
                cmd_refresh += ["--include-weekdays", scan_cfg["scan_include_weekdays"]]
            if scan_cfg["scan_exclude_weekdays"].strip():
                cmd_refresh += ["--exclude-weekdays", scan_cfg["scan_exclude_weekdays"]]
            cmd_fill = [
                "main.py",
                "--color",
                "never",
                "--season-code",
                season_code,
                "--odds-csv",
                ODDS_CSV,
                "--date",
                scan_date,
                "--auto-fill-odds",
                "--auto-fill-margin",
                str(auto_fill_margin),
                "--auto-fill-overwrite",
            ]
            if scan_cfg["scan_include_weekdays"].strip():
                cmd_fill += ["--include-weekdays", scan_cfg["scan_include_weekdays"]]
            if scan_cfg["scan_exclude_weekdays"].strip():
                cmd_fill += ["--exclude-weekdays", scan_cfg["scan_exclude_weekdays"]]
            cmd_scan = build_scan_cmd(scan_cfg)
            rc1, out1 = run_cmd(cmd_refresh)
            rc2, out2 = (1, "")
            rc3, out3 = (1, "")
            if rc1 == 0:
                rc2, out2 = run_cmd(cmd_fill)
            if rc1 == 0 and rc2 == 0:
                rc3, out3 = run_cmd(cmd_scan)
                st.session_state.last_scan_output = out3
                st.session_state.last_scan_cmd = " ".join(cmd_scan)

            st.code(
                "\n\n".join(
                    [
                        f"$ python {' '.join(cmd_refresh)}\n{out1}",
                        f"$ python {' '.join(cmd_fill)}\n{out2}",
                        f"$ python {' '.join(cmd_scan)}\n{out3}",
                    ]
                ),
                language="bash",
            )
            if rc1 == 0 and rc2 == 0 and rc3 == 0:
                picks, shortlist = parse_scan_summary(out3)
                st.success("Pipeline complete")
                st.info(f"Qualifying picks: {picks} | Shortlist: {shortlist}")
            else:
                st.error("Pipeline failed")

    with b:
        if st.button("Run Scan Only", use_container_width=True):
            cmd = build_scan_cmd(scan_cfg)
            rc, out = run_cmd(cmd)
            if rc == 0:
                st.session_state.last_scan_output = out
                st.session_state.last_scan_cmd = " ".join(cmd)
            st.code(f"$ python {' '.join(cmd)}\n{out}", language="bash")
            if rc == 0:
                picks, shortlist = parse_scan_summary(out)
                st.success("Scan complete")
                st.info(f"Qualifying picks: {picks} | Shortlist: {shortlist}")
            else:
                st.error("Scan failed")

    if st.session_state.last_scan_output:
        st.markdown("---")
        st.subheader("Parsed Scan Tables")
        all_rows = parse_report_table(st.session_state.last_scan_output, "SNIPER REPORT (VALUE FOUND)")
        shortlist_rows = parse_report_table(st.session_state.last_scan_output, "PARLAY SHORTLIST")
        c_a, c_b = st.columns(2)
        c_a.metric("All Qualifying Picks", len(all_rows))
        c_b.metric("Shortlist Picks", len(shortlist_rows))

        if all_rows:
            st.caption("SNIPER REPORT (sortable)")
            st.dataframe(rows_to_df(all_rows, "Edge"), use_container_width=True, hide_index=True)
        else:
            st.info("No qualifying picks in the last scan output.")

        if shortlist_rows:
            st.caption("PARLAY SHORTLIST (sortable)")
            st.dataframe(rows_to_df(shortlist_rows, "Edge"), use_container_width=True, hide_index=True)
        else:
            st.info("No shortlist found in the last scan output.")

with tab_results:
    st.subheader("Settle and Evaluate")
    r1, r2, r3 = st.columns(3)
    with r1:
        settle_mode = st.selectbox("Settle Up To", ["Yesterday", "Today", "Custom date"], key="settle_date_mode")
        settle_custom = st.date_input("Settle Date (custom)", key="settle_date_custom", disabled=settle_mode != "Custom date")
        settle_date = date_from_choice(settle_mode, settle_custom)
    with r2:
        report_mode = st.selectbox("Results Date Filter", ["All", "Today", "Yesterday", "Custom date"], key="report_date_mode")
        report_custom = st.date_input("Results Date (custom)", key="report_date_custom", disabled=report_mode != "Custom date")
        report_date = date_from_choice(report_mode, report_custom, allow_all=True)
    with r3:
        report_limit = st.number_input("Report Limit", min_value=1, value=50, step=1, key="report_limit")

    p1, p2 = st.columns(2)
    with p1:
        if st.button("Settle + Show Results", use_container_width=True, type="primary"):
            settle_cmd = [
                "settle_results.py",
                "--bets-log-csv",
                BETS_CSV,
                "--season-code",
                season_code,
                "--up-to-date",
                settle_date,
            ]
            report_cmd = [
                "results_report.py",
                "--bets-log-csv",
                BETS_CSV,
                "--date",
                report_date,
                "--limit",
                str(report_limit),
                "--color",
                "never",
            ]
            rc1, out1 = run_cmd(settle_cmd)
            rc2, out2 = run_cmd(report_cmd)
            st.code(
                f"$ python {' '.join(settle_cmd)}\n{out1}\n\n$ python {' '.join(report_cmd)}\n{out2}",
                language="bash",
            )
            if rc1 == 0 and rc2 == 0:
                st.success("Settlement + report complete")
            else:
                st.error("One of the steps failed")

    with p2:
        if st.button("Show Results Only", use_container_width=True):
            cmd = [
                "results_report.py",
                "--bets-log-csv",
                BETS_CSV,
                "--date",
                report_date,
                "--limit",
                str(report_limit),
                "--color",
                "never",
            ]
            rc, out = run_cmd(cmd)
            st.code(f"$ python {' '.join(cmd)}\n{out}", language="bash")
            if rc == 0:
                st.success("Report complete")
            else:
                st.error("Report failed")

    st.markdown("---")
    st.subheader("Played Picks (Won/Lost)")
    pp1, pp2, pp3 = st.columns(3)
    with pp1:
        played_mode = st.selectbox("Played Date Filter", ["All", "Today", "Yesterday", "Custom date"], key="played_date_mode")
        played_custom = st.date_input("Played Date (custom)", key="played_date_custom", disabled=played_mode != "Custom date")
        played_date = date_from_choice(played_mode, played_custom, allow_all=True)
    with pp2:
        played_profile = st.selectbox("Played Profile Filter", ["all", "manual", "pro_live"], index=0, key="played_profile_filter")
    with pp3:
        if st.button("Refresh Played Picks", use_container_width=True):
            st.rerun()

    profile_filter = "" if played_profile == "all" else played_profile
    played_rows = load_played_picks(date_filter=played_date, profile_filter=profile_filter)
    if not played_rows:
        st.info("No settled picks found yet for this filter. Run settlement first, then refresh.")
    else:
        wins = sum(1 for r in played_rows if r["Result"] == "WIN")
        losses = sum(1 for r in played_rows if r["Result"] == "LOSS")
        pushes = sum(1 for r in played_rows if r["Result"] == "PUSH")
        pnl = sum(to_float(r["PnL U"]) for r in played_rows)
        risked = sum(max(0.0, to_float(r["StakeU"])) for r in played_rows)
        roi = (pnl / risked * 100.0) if risked > 0 else 0.0
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Settled Picks", len(played_rows))
        s2.metric("Wins", wins)
        s3.metric("Losses", losses)
        s4.metric("Pushes", pushes)
        s5.metric("ROI", f"{roi:+.2f}%")

        if pd is not None:
            df = pd.DataFrame(played_rows)
            df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["_pnl"] = pd.to_numeric(df["PnL U"].str.replace("+", "", regex=False), errors="coerce")
            df = df.sort_values(["_date", "_pnl"], ascending=[False, False]).drop(columns=["_date", "_pnl"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(played_rows, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Parlay Picks Result Check")
    pr1, pr2 = st.columns(2)
    with pr1:
        if st.button("Check Parlay Results", use_container_width=True, type="primary"):
            st.session_state["show_parlay_results"] = True
    with pr2:
        if st.button("Clear Parlay Results", use_container_width=True):
            st.session_state["show_parlay_results"] = False

    if st.session_state.get("show_parlay_results", False):
        parlay_rows = load_parlay_results(date_filter=played_date, profile_filter=profile_filter)
        parlay_leg_rows = load_parlay_leg_rows(date_filter=played_date, profile_filter=profile_filter)
        if not parlay_rows:
            st.info("No parlay tickets found (needs 2+ picks logged in one scan run).")
        else:
            won_t = sum(1 for r in parlay_rows if r["Parlay Result"] == "WON")
            lost_t = sum(1 for r in parlay_rows if r["Parlay Result"] == "LOST")
            push_t = sum(1 for r in parlay_rows if r["Parlay Result"] == "PUSH")
            pending_t = sum(1 for r in parlay_rows if r["Parlay Result"] == "PENDING")
            pnl_t = sum(to_float(r["Parlay PnL U (1u)"]) for r in parlay_rows if r["Parlay Result"] != "PENDING")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Parlay Tickets", len(parlay_rows))
            m2.metric("Won", won_t)
            m3.metric("Lost", lost_t)
            m4.metric("Pending", pending_t)
            m5.metric("PnL U (1u)", f"{pnl_t:+.2f}")
            if pd is not None:
                dfp = pd.DataFrame(parlay_rows)
                dfp["_date"] = pd.to_datetime(dfp["Date"], errors="coerce")
                dfp = dfp.sort_values(["_date", "Logged At"], ascending=[False, False]).drop(columns=["_date"])
                st.dataframe(dfp, use_container_width=True, hide_index=True)
            else:
                st.dataframe(parlay_rows, use_container_width=True, hide_index=True)

            st.caption("Parlay Legs (teams and markets)")
            if parlay_leg_rows:
                if pd is not None:
                    dfl = pd.DataFrame(parlay_leg_rows)
                    dfl["_date"] = pd.to_datetime(dfl["Date"], errors="coerce")
                    dfl = dfl.sort_values(["_date", "Ticket ID"], ascending=[False, False]).drop(columns=["_date"])
                    st.dataframe(dfl, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(parlay_leg_rows, use_container_width=True, hide_index=True)

with tab_backtest:
    st.subheader("Backtest Lab")
    st.caption("Test settings on historical data before using them in live scans.")
    st.info(
        "How to use: 1) choose seasons + markets, 2) run single backtest, "
        "3) run grid search to find stronger configs, 4) copy winning filters back to Scan."
    )
    bs1, bs2 = st.columns(2)
    with bs1:
        if st.button("Apply Safe Baseline", use_container_width=True):
            st.session_state.bt_season_codes = "2324,2425,2526"
            st.session_state.bt_markets = ["o25"]
            st.session_state.bt_train_ratio = 0.70
            st.session_state.bt_min_edge = 3.0
            st.session_state.bt_odds_min = 0.0
            st.session_state.bt_odds_max = 0.0
            st.session_state.bt_include_weekdays = ""
            st.session_state.bt_exclude_weekdays = ""
            st.session_state.bt_include_leagues = ""
            st.session_state.bt_exclude_leagues = ""
            st.rerun()
    with bs2:
        if st.button("Randomize Settings", use_container_width=True):
            pool = BACKTEST_MARKETS[:]
            random.shuffle(pool)
            take = random.randint(1, min(5, len(pool)))
            chosen = pool[:take]
            min_options = [0.0, 1.4, 1.6, 1.8, 2.0]
            max_options = [0.0, 2.2, 2.6, 3.2]
            o_min = random.choice(min_options)
            o_max = random.choice(max_options)
            if o_max != 0.0 and o_min > o_max:
                o_min, o_max = 0.0, o_max
            st.session_state.bt_markets = chosen
            st.session_state.bt_min_edge = random.choice([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            st.session_state.bt_train_ratio = random.choice([0.65, 0.70, 0.75, 0.80])
            st.session_state.bt_odds_min = o_min
            st.session_state.bt_odds_max = o_max
            st.session_state.bt_exclude_weekdays = random.choice(["", "tue,wed,thu", "mon,tue", ""])
            st.session_state.bt_include_weekdays = ""
            st.session_state.bt_include_leagues = ""
            st.session_state.bt_exclude_leagues = ""
            st.rerun()

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        bt_season_codes = st.text_input("Season Codes", key="bt_season_codes", help="comma list for robust tests")
    with b2:
        bt_train_ratio = st.number_input("Train Ratio", min_value=0.5, max_value=0.9, step=0.01, key="bt_train_ratio")
    with b3:
        bt_min_edge = st.number_input("Min Edge %", min_value=0.0, max_value=50.0, step=0.5, key="bt_min_edge")
    with b4:
        bt_markets = st.multiselect(
            "Markets",
            options=BACKTEST_MARKETS,
            default=st.session_state.bt_markets,
            format_func=lambda x: f"{x} - {dict(MARKET_CHOICES).get(x, x)}",
            key="bt_markets",
        )

    b5, b6, b7, b8 = st.columns(4)
    with b5:
        bt_odds_min = st.number_input("Odds Min", min_value=0.0, max_value=20.0, step=0.1, key="bt_odds_min")
    with b6:
        bt_odds_max = st.number_input("Odds Max", min_value=0.0, max_value=20.0, step=0.1, key="bt_odds_max")
    with b7:
        bt_include_weekdays = st.text_input("Include Weekdays", key="bt_include_weekdays", help="ex: sat,sun")
    with b8:
        bt_exclude_weekdays = st.text_input("Exclude Weekdays", key="bt_exclude_weekdays", help="ex: tue,wed,thu")

    b9, b10 = st.columns(2)
    with b9:
        bt_include_leagues = st.text_input("Include Leagues", key="bt_include_leagues", help="comma league names")
    with b10:
        bt_exclude_leagues = st.text_input("Exclude Leagues", key="bt_exclude_leagues", help="comma league names")

    st.markdown("**Grid Search Options**")
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1:
        bt_grid_min_edges = st.text_input("Grid Min Edges", key="bt_grid_min_edges")
    with g2:
        bt_grid_odds_min = st.text_input("Grid Odds Min", key="bt_grid_odds_min")
    with g3:
        bt_grid_odds_max = st.text_input("Grid Odds Max", key="bt_grid_odds_max")
    with g4:
        bt_grid_min_bets = st.number_input("Grid Min Bets/Season", min_value=10, step=10, key="bt_grid_min_bets")
    with g5:
        bt_grid_top = st.number_input("Grid Top N", min_value=1, step=1, key="bt_grid_top")

    st.markdown("**Auto Random Search**")
    a1, a2 = st.columns(2)
    with a1:
        bt_auto_tries = st.number_input("Max Random Tries", min_value=5, max_value=300, step=5, key="bt_auto_tries")
    with a2:
        bt_auto_min_bets = st.number_input("Auto Min Bets", min_value=10, max_value=500, step=10, key="bt_auto_min_bets")

    rb1, rb2, rb3 = st.columns(3)
    with rb1:
        if st.button("Run Backtest", use_container_width=True, type="primary"):
            mkts = ",".join(bt_markets) if bt_markets else "o25"
            cmd = [
                "backtest.py",
                "--season-codes",
                bt_season_codes,
                "--train-ratio",
                str(bt_train_ratio),
                "--min-edge",
                str(bt_min_edge),
                "--markets",
                mkts,
                "--odds-min",
                str(bt_odds_min),
                "--odds-max",
                str(bt_odds_max),
            ]
            if bt_include_leagues.strip():
                cmd += ["--include-leagues", bt_include_leagues.strip()]
            if bt_exclude_leagues.strip():
                cmd += ["--exclude-leagues", bt_exclude_leagues.strip()]
            if bt_include_weekdays.strip():
                cmd += ["--include-weekdays", bt_include_weekdays.strip()]
            if bt_exclude_weekdays.strip():
                cmd += ["--exclude-weekdays", bt_exclude_weekdays.strip()]
            rc, out = run_cmd(cmd)
            st.code(f"$ python {' '.join(cmd)}\n{out}", language="bash")
            if rc == 0:
                st.success("Backtest complete")
            else:
                st.error("Backtest failed")

    with rb2:
        if st.button("Run Grid Search", use_container_width=True):
            mkts = ",".join(bt_markets) if bt_markets else "o25"
            cmd = [
                "backtest.py",
                "--season-codes",
                bt_season_codes,
                "--train-ratio",
                str(bt_train_ratio),
                "--markets",
                mkts,
                "--grid-search",
                "--grid-min-edges",
                bt_grid_min_edges,
                "--grid-odds-min",
                bt_grid_odds_min,
                "--grid-odds-max",
                bt_grid_odds_max,
                "--grid-min-bets-per-season",
                str(bt_grid_min_bets),
                "--grid-top",
                str(bt_grid_top),
            ]
            if bt_include_leagues.strip():
                cmd += ["--include-leagues", bt_include_leagues.strip()]
            if bt_exclude_leagues.strip():
                cmd += ["--exclude-leagues", bt_exclude_leagues.strip()]
            if bt_include_weekdays.strip():
                cmd += ["--include-weekdays", bt_include_weekdays.strip()]
            if bt_exclude_weekdays.strip():
                cmd += ["--exclude-weekdays", bt_exclude_weekdays.strip()]
            rc, out = run_cmd(cmd)
            st.code(f"$ python {' '.join(cmd)}\n{out}", language="bash")
            if rc == 0:
                st.success("Grid search complete")
                grid_rows = parse_grid_rows(out, int(bt_grid_min_bets))
                if grid_rows:
                    st.caption("Grid rows with deploy-candidate label")
                    if pd is not None:
                        dfg = pd.DataFrame(grid_rows)
                        st.dataframe(dfg, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(grid_rows, use_container_width=True, hide_index=True)
                    yes_count = sum(1 for r in grid_rows if r.get("deploy_candidate") == "YES")
                    st.info(
                        f"Deploy candidates: {yes_count}/{len(grid_rows)} "
                        f"(rule: ROI > 0 and min bets >= {int(bt_grid_min_bets)})."
                    )
            else:
                st.error("Grid search failed")

    with rb3:
        if st.button("Auto-Find Profitable", use_container_width=True):
            market_pool = BACKTEST_MARKETS[:]
            best_cfg: Dict[str, object] | None = None
            best_roi = -999.0
            target_bets = int(bt_auto_min_bets)
            tries = int(bt_auto_tries)
            progress = st.progress(0)
            log_lines: List[str] = []
            found = False
            found_cfg: Dict[str, object] | None = None
            for i in range(tries):
                random.shuffle(market_pool)
                take = random.randint(1, min(4, len(market_pool)))
                mkts = market_pool[:take]
                edge = random.choice([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                tr = random.choice([0.65, 0.70, 0.75, 0.80])
                o_min = random.choice([0.0, 1.4, 1.6, 1.8, 2.0])
                o_max = random.choice([0.0, 2.2, 2.6, 3.2, 4.0])
                if o_max != 0.0 and o_min > o_max:
                    o_min = 0.0
                ex_wd = random.choice(["", "tue,wed,thu", "mon,tue", ""])
                season_list = [s.strip() for s in bt_season_codes.split(",") if s.strip()]
                season_code = random.choice(season_list) if season_list else "2526"
                cmd = [
                    "backtest.py",
                    "--season-code",
                    season_code,
                    "--train-ratio",
                    str(tr),
                    "--min-edge",
                    str(edge),
                    "--markets",
                    ",".join(mkts),
                    "--odds-min",
                    str(o_min),
                    "--odds-max",
                    str(o_max),
                ]
                if ex_wd:
                    cmd += ["--exclude-weekdays", ex_wd]

                rc, out = run_cmd(cmd)
                metrics = parse_single_backtest_metrics(out)
                roi = float(metrics["roi"])
                bets = int(metrics["bets"])
                if roi > best_roi and bets >= target_bets:
                    best_roi = roi
                    best_cfg = {
                        "season_code": season_code,
                        "train_ratio": tr,
                        "min_edge": edge,
                        "markets": mkts,
                        "odds_min": o_min,
                        "odds_max": o_max,
                        "exclude_weekdays": ex_wd,
                        "bets": bets,
                        "roi": roi,
                    }
                log_lines.append(
                    f"try {i+1}/{tries} | season={season_code} mkts={','.join(mkts)} edge={edge:.1f} "
                    f"odds=[{o_min:.1f},{o_max:.1f}] ex_wd={ex_wd or '-'} -> bets={bets} roi={roi:.2f}%"
                )
                progress.progress(min(100, int(((i + 1) / tries) * 100)))
                if rc == 0 and bets >= target_bets and roi > 0.0:
                    found = True
                    found_cfg = {
                        "season_code": season_code,
                        "train_ratio": tr,
                        "min_edge": edge,
                        "markets": mkts,
                        "odds_min": o_min,
                        "odds_max": o_max,
                        "exclude_weekdays": ex_wd,
                        "bets": bets,
                        "roi": roi,
                    }
                    break

            st.code("\n".join(log_lines[-min(len(log_lines), 50) :]), language="text")
            if found and found_cfg:
                st.success(
                    "Found profitable config: "
                    f"ROI={found_cfg['roi']:.2f}% | bets={found_cfg['bets']} | "
                    f"markets={','.join(found_cfg['markets'])}"
                )
                st.json(found_cfg)
            else:
                st.warning("No profitable config found in this random search window.")
                if best_cfg:
                    st.info(
                        "Best non-profitable (or near) config found: "
                        f"ROI={best_cfg['roi']:.2f}% | bets={best_cfg['bets']} | markets={','.join(best_cfg['markets'])}"
                    )
                    st.json(best_cfg)

with tab_logs:
    st.subheader("Files and Help")
    st.write(f"- Odds file: `{os.path.join(BASE_DIR, ODDS_CSV)}`")
    st.write(f"- Bets log: `{os.path.join(BASE_DIR, BETS_CSV)}`")
    st.write("- Tip: run `Run Daily Pipeline` first, then settle next day.")
    if os.path.exists(os.path.join(BASE_DIR, BETS_CSV)):
        with st.expander("Preview bets_log.csv (first 30 rows)", expanded=False):
            with open(os.path.join(BASE_DIR, BETS_CSV), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))[:30]
            st.dataframe(rows, use_container_width=True)

st.caption(f"Today: {date.today().isoformat()} | Project: {BASE_DIR}")
