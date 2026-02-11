from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from datetime import date
from typing import Dict, List, Tuple

import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ODDS_CSV = "upcoming_odds.csv"
BETS_CSV = "bets_log.csv"


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

st.title("Soccer Bot Free")
st.caption("Scanner + settlement control panel")

with st.sidebar:
    st.subheader("Scan Settings")
    season_code = st.text_input("Season Code", value="2526")
    scan_date = st.text_input("Scan Date", value="today", help="today, tomorrow, or YYYY-MM-DD")
    mode = st.selectbox("Mode", ["safe", "balanced", "aggressive"], index=2)
    min_edge = st.number_input("Min Edge %", value=8.0, step=0.1)
    max_picks = st.number_input("Max Picks", min_value=0, value=10, step=1)
    one_pick_per_match = st.checkbox("One Pick Per Match", value=True)
    shortlist_market_cap = st.number_input("Shortlist Market Cap", min_value=0, value=3, step=1)
    log_picks = st.checkbox("Log Picks", value=True)
    bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0, step=10.0)
    profile = st.selectbox("Profile", ["none", "pro_live"], index=0)

    with st.expander("Advanced Filters", expanded=False):
        odds_min = st.number_input("Odds Min", min_value=0.0, value=0.0, step=0.1)
        odds_max = st.number_input("Odds Max", min_value=0.0, value=0.0, step=0.1)
        include_markets = st.text_input("Include Markets", value="", help="comma keys: o25,dnb,home,away")
        exclude_markets = st.text_input("Exclude Markets", value="", help="comma keys: co105,ahc_h_m2_5")
        refresh_lookahead_days = st.number_input("Refresh Lookahead Days", min_value=0, value=7, step=1)
        auto_fill_margin = st.number_input("Auto-fill Margin %", value=-8.0, step=0.5)

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
    "include_markets": include_markets,
    "exclude_markets": exclude_markets,
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
            st.code(f"$ python {' '.join(cmd)}\n{out}", language="bash")
            if rc == 0:
                picks, shortlist = parse_scan_summary(out)
                st.success("Scan complete")
                st.info(f"Qualifying picks: {picks} | Shortlist: {shortlist}")
            else:
                st.error("Scan failed")

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
