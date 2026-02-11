from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from datetime import date
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
if "last_scan_output" not in st.session_state:
    st.session_state.last_scan_output = ""
if "last_scan_cmd" not in st.session_state:
    st.session_state.last_scan_cmd = ""

st.title("Soccer Bot Free")
st.caption("Scanner + settlement control panel")

with st.sidebar:
    st.subheader("Scan Settings")
    preset = st.selectbox(
        "Preset",
        ["DailyScan Morning (11 Feb style)", "Balanced", "Conservative", "Custom"],
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
            st.session_state.scan_date = "today"
            st.session_state.log_picks = True
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
            st.session_state.scan_date = "today"
            st.session_state.log_picks = True
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
            st.session_state.scan_date = "today"
            st.session_state.log_picks = True
        st.rerun()
    season_code = st.text_input("Season Code", value="2526")
    scan_date = st.text_input("Scan Date", key="scan_date", help="today, tomorrow, or YYYY-MM-DD")
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
}

tab_scan, tab_results, tab_logs = st.tabs(["Scan", "Results", "Files"])

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
        settle_date = st.text_input("Settle Up To", value="yesterday", key="settle_date")
    with r2:
        report_date = st.text_input("Results Date Filter", value="all", key="report_date")
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
