from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, timedelta
from typing import Dict, List


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"


def should_use_color(color_mode: str) -> bool:
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{ANSI_RESET}"


def resolve_date(raw: str) -> str:
    token = (raw or "").strip().lower()
    today = date.today()
    if token in {"", "all"}:
        return "all"
    if token == "today":
        return today.isoformat()
    if token == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    if token == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    return raw.strip()


def to_float(raw: str) -> float:
    try:
        return float((raw or "").strip())
    except (TypeError, ValueError):
        return 0.0


def is_settled(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "y"}


def classify_row(r: Dict[str, str]) -> str:
    if not is_settled(r.get("settled", "")):
        return "PENDING"
    result = (r.get("result") or "").strip().lower()
    if result in {"win", "won", "w"}:
        return "WON"
    if result in {"loss", "lost", "l"}:
        return "LOST"
    if result in {"push", "void", "draw", "p"}:
        return "PUSH"
    pnl = to_float(r.get("pnl_units", "0"))
    if pnl > 1e-9:
        return "WON"
    if pnl < -1e-9:
        return "LOST"
    return "PUSH"


def fmt_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))


def print_table(lines: List[List[str]]) -> str:
    if not lines:
        return ""
    widths = [len(x) for x in lines[0]]
    for r in lines[1:]:
        for i, x in enumerate(r):
            widths[i] = max(widths[i], len(x))
    out = [fmt_row(lines[0], widths), "-" * (sum(widths) + (3 * (len(widths) - 1)))]
    for r in lines[1:]:
        out.append(fmt_row(r, widths))
    return "\n".join(out)


def build_section_rows(items: List[Dict[str, str]], limit: int) -> List[List[str]]:
    rows = [["Date", "League", "Fixture", "Market", "StakeU", "PnL U", "Edge", "Conf"]]
    for r in items[:limit]:
        rows.append(
            [
                (r.get("date") or "N/A").strip(),
                (r.get("league") or "N/A").strip(),
                (r.get("fixture") or "").strip(),
                (r.get("market") or "").strip(),
                f"{to_float(r.get('stake_units', '0')):.2f}",
                f"{to_float(r.get('pnl_units', '0')):+.2f}",
                f"{to_float(r.get('edge_pct', '0')):.1f}%",
                f"{to_float(r.get('confidence_pct', '0')):.1f}%",
            ]
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show won/lost/pending results from bets log")
    p.add_argument("--bets-log-csv", default="bets_log.csv", help="Path to bets log CSV")
    p.add_argument("--date", default="all", help="all, today, yesterday, tomorrow, or YYYY-MM-DD")
    p.add_argument("--profile", default="", help="Optional profile filter (e.g. pro_live)")
    p.add_argument("--limit", type=int, default=20, help="Max rows per section")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto", help="Color output mode")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    use_color = should_use_color(args.color)
    target_date = resolve_date(args.date)

    if not os.path.exists(args.bets_log_csv):
        print(f"Bets log not found: {args.bets_log_csv}")
        return 1

    with open(args.bets_log_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.profile:
        rows = [r for r in rows if (r.get("profile") or "").strip() == args.profile.strip()]
    if target_date != "all":
        rows = [r for r in rows if (r.get("date") or "").strip() == target_date]

    if not rows:
        print("No bets found for selected filters.")
        return 0

    won: List[Dict[str, str]] = []
    lost: List[Dict[str, str]] = []
    push: List[Dict[str, str]] = []
    pending: List[Dict[str, str]] = []
    settled = 0
    risked_u = 0.0
    pnl_u = 0.0
    for r in rows:
        status = classify_row(r)
        if status == "PENDING":
            pending.append(r)
            continue
        settled += 1
        risked_u += max(0.0, to_float(r.get("stake_units", "0")))
        pnl_u += to_float(r.get("pnl_units", "0"))
        if status == "WON":
            won.append(r)
        elif status == "LOST":
            lost.append(r)
        else:
            push.append(r)

    roi = (pnl_u / risked_u * 100.0) if risked_u > 0 else 0.0

    title = c("BET RESULTS DASHBOARD", ANSI_BOLD + ANSI_CYAN, use_color)
    print(title)
    print(c(f"Filter date: {target_date}", ANSI_DIM, use_color))
    if args.profile:
        print(c(f"Filter profile: {args.profile}", ANSI_DIM, use_color))
    print(
        f"Total: {len(rows)} | Settled: {settled} | "
        f"{c('Won', ANSI_GREEN, use_color)}: {len(won)} | "
        f"{c('Lost', ANSI_RED, use_color)}: {len(lost)} | "
        f"{c('Push', ANSI_YELLOW, use_color)}: {len(push)} | "
        f"{c('Pending', ANSI_YELLOW, use_color)}: {len(pending)}"
    )
    print(f"Settled PnL: {pnl_u:+.2f}u | ROI: {roi:+.2f}%")
    print("")

    if won:
        print(c(f"WON ({len(won)})", ANSI_BOLD + ANSI_GREEN, use_color))
        print(print_table(build_section_rows(won, args.limit)))
        print("")
    if lost:
        print(c(f"LOST ({len(lost)})", ANSI_BOLD + ANSI_RED, use_color))
        print(print_table(build_section_rows(lost, args.limit)))
        print("")
    if push:
        print(c(f"PUSH ({len(push)})", ANSI_BOLD + ANSI_YELLOW, use_color))
        print(print_table(build_section_rows(push, args.limit)))
        print("")
    if pending:
        print(c(f"PENDING ({len(pending)})", ANSI_BOLD + ANSI_YELLOW, use_color))
        print(print_table(build_section_rows(pending, args.limit)))
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
