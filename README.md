# Soccer Bot (Free / No API Keys)

This is a keyless version of your report's bot:
- Uses free historical data from football-data.co.uk
- Uses a local CSV for bookmaker odds (`upcoming_odds.csv`)
- Runs Poisson fair-odds checks for:
  - Straight wins (home/away) + DNB (home/away)
  - Asian handicap (goals and corners)
  - Goals overs/unders (1.5/2.5/3.5)
  - BTTS Yes/No
  - Corners overs/unders (8.5/9.5/10.5)
- Uses only Python standard library (no pip install needed)

## Quick Start

```bash
cd "/Users/adebara/Documents/New project/soccer_bot_free"
cp upcoming_odds.example.csv upcoming_odds.csv
python3 main.py --odds-csv upcoming_odds.csv --season-code 2526
```

Auto-import tomorrow fixtures (keyless) then scan:

```bash
python3 main.py --season-code 2526 --odds-csv upcoming_odds.csv --refresh-odds --refresh-date tomorrow --refresh-lookahead-days 5 --date tomorrow
```

If bookmaker-odds feeds are unavailable for that date, refresh falls back to fixture-only rows (schedule still loads, odds columns stay blank until you fill them).

Default behavior scans only today's fixtures (`--date today`).
Use `--all-games` to scan every fixture in the CSV.
Each run writes a text report to `reports/`.
For live scanning, add `--live-only` and optionally `--watch --interval 30`.
To list fixtures missing key odds fields and export a fill template:

```bash
python3 main.py --odds-csv upcoming_odds.csv --date 2026-02-12 --show-missing-odds --export-missing-template missing_odds_fill.csv
```

After filling `missing_odds_fill.csv`, merge it into `upcoming_odds.csv`:

```bash
python3 main.py --odds-csv upcoming_odds.csv --merge-filled-odds missing_odds_fill.csv
```

If you do not want manual entry, auto-fill missing odds from model estimates:

```bash
python3 main.py --season-code 2526 --odds-csv upcoming_odds.csv --date 2026-02-12 --auto-fill-odds --auto-fill-margin 6
```

To keep reports compact:

```bash
python3 main.py --season-code 2526 --odds-csv upcoming_odds.csv --date 2026-02-12 --mode aggressive --one-pick-per-match --max-picks 10
```

This keeps the full qualifying market report, and adds a bottom "Parlay Shortlist (Top N)" section.

Optional market filtering (default remains all markets):

```bash
python3 main.py --season-code 2526 --odds-csv upcoming_odds.csv --date 2026-02-12 --mode aggressive --include-markets o25,dnb,home,away --exclude-markets co105,ahc_h_m2_5
```

Optional shortlist diversification cap (default off):

```bash
python3 main.py --season-code 2526 --odds-csv upcoming_odds.csv --date 2026-02-12 --mode aggressive --max-picks 10 --one-pick-per-match --shortlist-market-cap 2
```

Market keys:
- `home`, `away`, `dnb`, `adnb`
- `ah_h_m0_5`, `ah_a_p0_5`, `ah_h_m1_5`, `ah_a_p1_5`
- `ahc_h_m1_5`, `ahc_a_p1_5`, `ahc_h_m2_5`, `ahc_a_p2_5`
- `o15`, `o25`, `o35`, `u25`, `u35`
- `btts_yes`, `btts_no`
- `co85`, `co95`, `co105`, `cu95`, `cu105`

Controlled live mode (no paper trading), with strict profile:

```bash
python3 main.py --season-code 2526 --odds-csv upcoming_odds.csv --date 2026-02-12 --profile pro_live --max-picks 10
```

`pro_live` enforces:
- market: `Over 2.5 Goals` only
- min edge: `9%`
- odds band: `2.0` to `3.2`
- one pick per match in shortlist
- flat stake: `0.25%` bankroll
- stop-loss guard from `bets_log.csv` (`-3u` day, `-20u` total)
- auto-log shortlist picks into `bets_log.csv`

Backtest on historical seasons (flat 1u staking):

```bash
python3 backtest.py --season-code 2526 --train-ratio 0.70 --min-edge 3 --markets home,o25
```

Backtest filters:

```bash
python3 backtest.py --season-code 2526 --train-ratio 0.70 --min-edge 6 --markets o25 --odds-min 1.8 --odds-max 2.6 --exclude-weekdays tue,wed,thu
```

Grid-search best settings:

```bash
python3 backtest.py --season-code 2526 --train-ratio 0.70 --markets o25 --grid-search --grid-min-edges 3,5,7,9 --grid-odds-min 1.6,1.8,2.0 --grid-odds-max 2.2,2.6,3.2 --grid-min-bets 80 --grid-top 10
```

Robust multi-season grid-search:

```bash
python3 backtest.py --season-codes 2324,2425,2526 --train-ratio 0.70 --markets all --grid-search --grid-min-edges 3,5,7,9 --grid-odds-min 1.6,1.8,2.0 --grid-odds-max 2.2,2.6,3.2 --grid-min-bets-per-season 60 --grid-top 10
```

Per-market robust leaderboard (best config per market):

```bash
python3 backtest.py --season-codes 2324,2425,2526 --grid-search --market-leaderboard --train-ratio 0.70 --grid-min-edges 3,5,7,9 --grid-odds-min 1.6,1.8,2.0 --grid-odds-max 2.2,2.6,3.2 --grid-min-bets-per-season 60 --grid-top 14
```

Or double-click `DailyScan.command` on macOS.

## Streamlit Web App

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy on Streamlit Community Cloud:

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. New app:
   - Repository: your repo
   - Branch: `main`
   - Main file path: `app.py`
4. Deploy.

The app includes:
- Daily pipeline run (refresh -> auto-fill -> scan)
- Scan-only mode
- Results dashboard view
- Auto-settle + results view

## Edit Odds Input

Fill `upcoming_odds.csv` with your matches and current odds:

```csv
date,league,home_team,away_team,is_live,minute,home_win_odds,away_win_odds,home_dnb_odds,away_dnb_odds,ah_home_m0_5_odds,ah_away_p0_5_odds,ah_home_m1_5_odds,ah_away_p1_5_odds,ahc_home_m1_5_odds,ahc_away_p1_5_odds,ahc_home_m2_5_odds,ahc_away_p2_5_odds,over_1_5_goals_odds,over_2_5_goals_odds,over_3_5_goals_odds,under_2_5_goals_odds,under_3_5_goals_odds,btts_yes_odds,btts_no_odds,over_8_5_corners_odds,over_9_5_corners_odds,over_10_5_corners_odds,under_9_5_corners_odds,under_10_5_corners_odds
2026-02-14,EPL,Arsenal,Chelsea,0,0,2.15,3.30,1.62,2.20,2.15,1.70,3.40,1.35,2.00,1.80,2.40,1.55,1.28,1.92,3.10,1.88,1.36,1.72,2.05,1.55,1.95,2.40,1.80,1.52
```

You can include only the odds columns you use. Any supported `*_odds` column is picked up automatically.

League handling:
- Accepts almost any league name from CSV.
- Normalizes common aliases (for example: `EPL`, `Premier League`, `Portugal`, `MLS`, `Liga MX`).
- Uses league-specific model when available; otherwise falls back to a global model.

## Notes

- No API key is required.
- Since there is no live odds API, you paste odds manually.
- Season code format follows football-data.co.uk folder style (example: `2526`).
- If you want fully offline runs, provide local historical CSVs and use `--local-data-dir`.

Offline data file names expected:
- `E0.csv` (EPL)
- `E1.csv` (Championship)
- `E2.csv` (LeagueOne)
- `E3.csv` (LeagueTwo)
- `SP1.csv` (LaLiga)
- `SP2.csv` (LaLiga2)
- `D1.csv` (Bundesliga)
- `D2.csv` (Bundesliga2)
- `I1.csv` (SerieA)
- `I2.csv` (SerieB)
- `F1.csv` (Ligue1)
- `F2.csv` (Ligue2)
- `N1.csv` (Eredivisie)
- `P1.csv` (PrimeiraLiga)
- `B1.csv` (BelgiumProLeague)
- `T1.csv` (TurkeySuperLig)
- `SC0.csv` (ScotlandPremiership)
- `A1.csv` (AustriaBundesliga)
- `DNK.csv` (DenmarkSuperliga)
- `SWE.csv` (SwedenAllsvenskan)
- `NOR.csv` (NorwayEliteserien)
