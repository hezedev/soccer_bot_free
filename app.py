from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from typing import List

import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run_cmd(args: List[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [PY] + args,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


st.set_page_config(page_title="Soccer Bot Free", page_icon="⚽", layout="wide")
st.title("Soccer Bot Free")
st.caption("Keyless scanner dashboard")

if not os.path.exists(os.path.join(BASE_DIR, "upcoming_odds.csv")):
    example = os.path.join(BASE_DIR, "upcoming_odds.example.csv")
    if os.path.exists(example):
        with open(example, "r", encoding="utf-8") as src, open(
            os.path.join(BASE_DIR, "upcoming_odds.csv"), "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

col1, col2, col3, col4 = st.columns(4)
with col1:
    season_code = st.text_input("Season Code", value="2526")
with col2:
    scan_date = st.text_input("Scan Date", value="today")
with col3:
    mode = st.selectbox("Mode", ["safe", "balanced", "aggressive"], index=2)
with col4:
    min_edge = st.number_input("Min Edge %", value=8.0, step=0.1)

col5, col6, col7, col8 = st.columns(4)
with col5:
    max_picks = st.number_input("Max Picks", min_value=0, value=10, step=1)
with col6:
    one_pick_per_match = st.checkbox("One Pick Per Match", value=True)
with col7:
    shortlist_market_cap = st.number_input("Shortlist Market Cap", min_value=0, value=3, step=1)
with col8:
    log_picks = st.checkbox("Log Picks", value=True)

st.markdown("---")

left, right = st.columns(2)
with left:
    if st.button("Run Daily Pipeline", use_container_width=True):
        cmds = [
            [
                "main.py",
                "--color",
                "never",
                "--season-code",
                season_code,
                "--odds-csv",
                "upcoming_odds.csv",
                "--refresh-odds",
                "--refresh-date",
                scan_date,
                "--refresh-lookahead-days",
                "7",
                "--date",
                scan_date,
            ],
            [
                "main.py",
                "--color",
                "never",
                "--season-code",
                season_code,
                "--odds-csv",
                "upcoming_odds.csv",
                "--date",
                scan_date,
                "--auto-fill-odds",
                "--auto-fill-margin",
                "-8",
                "--auto-fill-overwrite",
            ],
            [
                "main.py",
                "--color",
                "never",
                "--season-code",
                season_code,
                "--odds-csv",
                "upcoming_odds.csv",
                "--date",
                scan_date,
                "--mode",
                mode,
                "--min-edge",
                str(min_edge),
                "--max-picks",
                str(max_picks),
                "--shortlist-market-cap",
                str(shortlist_market_cap),
            ],
        ]
        if one_pick_per_match:
            cmds[-1].append("--one-pick-per-match")
        if log_picks:
            cmds[-1].append("--log-picks")

        full_out: List[str] = []
        code = 0
        for c in cmds:
            rc, out = run_cmd(c)
            code = rc
            full_out.append(f"$ python {' '.join(c)}\n{out}")
            if rc != 0:
                break
        st.code("\n\n".join(full_out), language="bash")
        if code == 0:
            st.success("Pipeline complete")
        else:
            st.error("Pipeline failed")

with right:
    if st.button("Run Scan Only", use_container_width=True):
        cmd = [
            "main.py",
            "--color",
            "never",
            "--season-code",
            season_code,
            "--odds-csv",
            "upcoming_odds.csv",
            "--date",
            scan_date,
            "--mode",
            mode,
            "--min-edge",
            str(min_edge),
            "--max-picks",
            str(max_picks),
            "--shortlist-market-cap",
            str(shortlist_market_cap),
        ]
        if one_pick_per_match:
            cmd.append("--one-pick-per-match")
        if log_picks:
            cmd.append("--log-picks")
        rc, out = run_cmd(cmd)
        st.code(f"$ python {' '.join(cmd)}\n{out}", language="bash")
        if rc == 0:
            st.success("Scan complete")
        else:
            st.error("Scan failed")

st.markdown("---")
st.subheader("Results")

res1, res2 = st.columns(2)
with res1:
    settle_date = st.text_input("Settle Up To", value="yesterday")
with res2:
    report_date = st.text_input("Results Date Filter", value="all")

res3, res4 = st.columns(2)
with res3:
    if st.button("Settle + Show Results", use_container_width=True):
        settle_cmd = [
            "settle_results.py",
            "--bets-log-csv",
            "bets_log.csv",
            "--season-code",
            season_code,
            "--up-to-date",
            settle_date,
        ]
        report_cmd = [
            "results_report.py",
            "--bets-log-csv",
            "bets_log.csv",
            "--date",
            report_date,
            "--limit",
            "50",
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

with res4:
    if st.button("Show Results Only", use_container_width=True):
        report_cmd = [
            "results_report.py",
            "--bets-log-csv",
            "bets_log.csv",
            "--date",
            report_date,
            "--limit",
            "50",
            "--color",
            "never",
        ]
        rc, out = run_cmd(report_cmd)
        st.code(f"$ python {' '.join(report_cmd)}\n{out}", language="bash")
        if rc == 0:
            st.success("Report complete")
        else:
            st.error("Report failed")

st.caption(f"Today: {date.today().isoformat()}")
