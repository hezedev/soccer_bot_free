from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def over_probability(line: float, lam_total: float) -> float:
    threshold = int(math.floor(line)) + 1
    p_under = 0.0
    for k in range(threshold):
        p_under += poisson_pmf(k, lam_total)
    return max(0.0, min(1.0, 1.0 - p_under))


@dataclass
class LeagueParams:
    avg_home_goals: float
    avg_away_goals: float
    avg_home_corners: float
    avg_away_corners: float
    attack_strength_goal: Dict[str, float]
    defense_strength_goal: Dict[str, float]
    attack_strength_corner: Dict[str, float]
    defense_strength_corner: Dict[str, float]


def _mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


class SimpleBettingModel:
    def __init__(self, shrink: float = 0.7):
        self.shrink = shrink
        self.params_by_league: Dict[str, LeagueParams] = {}

    def fit(self, rows: List[dict]) -> None:
        self.params_by_league = {}
        by_league: Dict[str, List[dict]] = {}
        for r in rows:
            by_league.setdefault(r["League"], []).append(r)

        for league, lrows in by_league.items():
            avg_home_goals = _mean([r["FTHG"] for r in lrows], 1.2)
            avg_away_goals = _mean([r["FTAG"] for r in lrows], 1.0)
            avg_home_corners = _mean([r["HC"] for r in lrows], 5.0)
            avg_away_corners = _mean([r["AC"] for r in lrows], 4.5)

            teams = sorted({r["HomeTeam"] for r in lrows} | {r["AwayTeam"] for r in lrows})
            attack_goal: Dict[str, float] = {}
            defense_goal: Dict[str, float] = {}
            attack_corner: Dict[str, float] = {}
            defense_corner: Dict[str, float] = {}

            for team in teams:
                gf: List[float] = []
                ga: List[float] = []
                cf: List[float] = []
                ca: List[float] = []
                for r in lrows:
                    if r["HomeTeam"] == team:
                        gf.append(r["FTHG"])
                        ga.append(r["FTAG"])
                        cf.append(r["HC"])
                        ca.append(r["AC"])
                    elif r["AwayTeam"] == team:
                        gf.append(r["FTAG"])
                        ga.append(r["FTHG"])
                        cf.append(r["AC"])
                        ca.append(r["HC"])

                base_goal = (avg_home_goals + avg_away_goals) / 2.0
                base_corner = (avg_home_corners + avg_away_corners) / 2.0

                raw_attack_goal = _mean(gf, base_goal) / base_goal if base_goal > 0 else 1.0
                raw_defense_goal = _mean(ga, base_goal) / base_goal if base_goal > 0 else 1.0
                raw_attack_corner = _mean(cf, base_corner) / base_corner if base_corner > 0 else 1.0
                raw_defense_corner = _mean(ca, base_corner) / base_corner if base_corner > 0 else 1.0

                attack_goal[team] = self.shrink * raw_attack_goal + (1 - self.shrink)
                defense_goal[team] = self.shrink * raw_defense_goal + (1 - self.shrink)
                attack_corner[team] = self.shrink * raw_attack_corner + (1 - self.shrink)
                defense_corner[team] = self.shrink * raw_defense_corner + (1 - self.shrink)

            self.params_by_league[league] = LeagueParams(
                avg_home_goals=avg_home_goals,
                avg_away_goals=avg_away_goals,
                avg_home_corners=avg_home_corners,
                avg_away_corners=avg_away_corners,
                attack_strength_goal=attack_goal,
                defense_strength_goal=defense_goal,
                attack_strength_corner=attack_corner,
                defense_strength_corner=defense_corner,
            )

    def _strength(self, m: Dict[str, float], team: str) -> float:
        return m.get(team, 1.0)

    def predict(self, league: str, home_team: str, away_team: str) -> Dict[str, float]:
        if league not in self.params_by_league:
            raise ValueError(f"League '{league}' has no fitted params.")
        p = self.params_by_league[league]

        home_goal_lam = (
            p.avg_home_goals
            * self._strength(p.attack_strength_goal, home_team)
            * self._strength(p.defense_strength_goal, away_team)
        )
        away_goal_lam = (
            p.avg_away_goals
            * self._strength(p.attack_strength_goal, away_team)
            * self._strength(p.defense_strength_goal, home_team)
        )
        home_corner_lam = (
            p.avg_home_corners
            * self._strength(p.attack_strength_corner, home_team)
            * self._strength(p.defense_strength_corner, away_team)
        )
        away_corner_lam = (
            p.avg_away_corners
            * self._strength(p.attack_strength_corner, away_team)
            * self._strength(p.defense_strength_corner, home_team)
        )
        return {
            "home_goal_lambda": max(0.1, home_goal_lam),
            "away_goal_lambda": max(0.1, away_goal_lam),
            "home_corner_lambda": max(0.1, home_corner_lam),
            "away_corner_lambda": max(0.1, away_corner_lam),
        }
