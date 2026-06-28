"""Environment-aware rollout budget allocation for SER."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class EnvironmentStats:
    name: str
    moving_reward: float = 0.0
    previous_reward: float = 0.0
    moving_cost_seconds: float = 1.0
    updates: int = 0
    last_reward: float = 0.0
    last_cost_seconds: float = 1.0

    def update(self, *, reward: float, cost_seconds: float, ema_alpha: float) -> None:
        self.previous_reward = self.moving_reward
        if self.updates == 0:
            self.moving_reward = reward
            self.moving_cost_seconds = max(cost_seconds, 1e-6)
        else:
            self.moving_reward = ema_alpha * reward + (1.0 - ema_alpha) * self.moving_reward
            self.moving_cost_seconds = (
                ema_alpha * max(cost_seconds, 1e-6)
                + (1.0 - ema_alpha) * self.moving_cost_seconds
            )
        self.last_reward = reward
        self.last_cost_seconds = max(cost_seconds, 1e-6)
        self.updates += 1

    def utility(self, floor: float, mode: str) -> float:
        if mode == "reward":
            if self.updates < 1:
                return floor
            return max(self.moving_reward, floor)
        if self.updates < 2:
            return floor
        return max(self.moving_reward - self.previous_reward, floor)

    def ratio(self, floor: float, cost_floor: float, mode: str) -> float:
        return self.utility(floor, mode) / max(self.moving_cost_seconds, cost_floor)


class BudgetAllocator:
    def __init__(
        self,
        env_names: list[str],
        *,
        ema_alpha: float = 0.2,
        utility_floor: float = 0.01,
        cost_floor: float = 1e-3,
        min_probability: float = 0.1,
        utility_mode: str = "reward",
        seed: int = 42,
    ) -> None:
        self.stats = {name: EnvironmentStats(name=name) for name in env_names}
        self.ema_alpha = ema_alpha
        self.utility_floor = utility_floor
        self.cost_floor = cost_floor
        self.min_probability = min_probability
        if utility_mode not in {"reward", "gain"}:
            raise ValueError("utility_mode must be 'reward' or 'gain'.")
        self.utility_mode = utility_mode
        self.rng = random.Random(seed)

    def probabilities(self) -> dict[str, float]:
        names = list(self.stats)
        raw = {
            name: self.stats[name].ratio(self.utility_floor, self.cost_floor, self.utility_mode)
            for name in names
        }
        total = sum(raw.values())
        if total <= 0:
            probs = {name: 1.0 / len(names) for name in names}
        else:
            probs = {name: value / total for name, value in raw.items()}

        if self.min_probability > 0:
            floor = min(self.min_probability, 1.0 / len(names))
            remaining = 1.0 - floor * len(names)
            if remaining <= 0:
                probs = {name: 1.0 / len(names) for name in names}
            else:
                probs = {
                    name: floor + remaining * probs[name]
                    for name in names
                }
        return normalize(probs)

    def select(self) -> str:
        probs = self.probabilities()
        names = list(probs)
        values = [probs[name] for name in names]
        return self.rng.choices(names, weights=values, k=1)[0]

    def weight(self, env_name: str) -> float:
        return self.probabilities().get(env_name, 0.0)

    def allocation(self, total_budget: int, *, ensure_each: bool = True) -> dict[str, int]:
        """Convert environment probabilities into integer sample counts."""

        total_budget = int(total_budget)
        if total_budget <= 0:
            raise ValueError("total_budget must be positive.")

        probs = self.probabilities()
        names = list(probs)
        counts = {name: 0 for name in names}
        remaining = total_budget

        if ensure_each and total_budget >= len(names):
            counts = {name: 1 for name in names}
            remaining -= len(names)

        quotas = {name: probs[name] * remaining for name in names}
        for name, quota in quotas.items():
            add = int(quota)
            counts[name] += add
            remaining -= add

        if remaining > 0:
            ranked = sorted(
                names,
                key=lambda name: (quotas[name] - int(quotas[name]), probs[name], name),
                reverse=True,
            )
            for name in ranked[:remaining]:
                counts[name] += 1

        return counts

    def update(self, env_name: str, *, reward: float, cost_seconds: float) -> None:
        self.stats[env_name].update(
            reward=reward,
            cost_seconds=cost_seconds,
            ema_alpha=self.ema_alpha,
        )

    def as_dict(self) -> dict[str, float]:
        logs: dict[str, float] = {}
        probs = self.probabilities()
        for name, stats in self.stats.items():
            logs[f"budget/{name}_probability"] = probs[name]
            logs[f"budget/{name}_moving_reward"] = stats.moving_reward
            logs[f"budget/{name}_moving_cost_seconds"] = stats.moving_cost_seconds
            logs[f"budget/{name}_ratio"] = stats.ratio(self.utility_floor, self.cost_floor, self.utility_mode)
            logs[f"budget/{name}_updates"] = float(stats.updates)
        return logs


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}
