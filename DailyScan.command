#!/bin/zsh
cd "$(dirname "$0")"
if [ ! -f upcoming_odds.csv ]; then
  cp upcoming_odds.example.csv upcoming_odds.csv
fi
python3 main.py --odds-csv upcoming_odds.csv --season-code 2526 --refresh-odds --refresh-date tomorrow --refresh-lookahead-days 5 --date tomorrow
read -k1 -r "?Press any key to close..."
