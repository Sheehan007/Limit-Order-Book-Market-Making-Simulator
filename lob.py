"""
Limit Order Book (LOB) Market Making Simulator
- Discrete-event top-of-book LOB with queues, limit arrivals, cancellations, and market orders
- Inventory-aware + adverse-selection-aware market making policy (Avellaneda-Stoikov-inspired)
- PnL decomposition: spread capture, inventory revaluation, adverse selection (effective vs realized spread)
- Regime evaluation (low/high volatility) and key plots

Dependencies: numpy, pandas, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Side(Enum):
    BID = 1
    ASK = -1


class EventType(Enum):
    LIMIT = auto()
    CANCEL = auto()
    MARKET = auto()


@dataclass(frozen=True)
class RegimeParams:
    name: str
    day_seconds: float = 6.5 * 60 * 60
    tick_size: float = 0.01
    levels: int = 10
    base_price: float = 100.0
    base_depth: float = 900.0
    depth_decay: float = 0.30

    lambda_limit: float = 1.2
    lambda_cancel: float = 0.7
    lambda_market: float = 0.5

    limit_size_mean: float = 80.0
    market_size_mean: float = 120.0
    cancel_max_frac: float = 0.35

    ofi_sensitivity: float = 0.20  # OFI -> P(MO buy) shift


@dataclass(frozen=True)
class StrategyParams:
    quote_interval_s: float = 0.25
    mm_order_size: float = 80.0
    max_pos: float = 900.0

    gamma: float = 0.10
    k: float = 1.2
    terminal_horizon_s: float = 60.0

    ofi_drift_coeff: float = 1.2
    toxicity_spread_coeff: float = 3.0
    max_quote_levels_away: int = 8


@dataclass
class Trade:
    event_idx: int
    time_s: float
    q_initiator: int  # +1 buy MO hits ask; -1 sell MO hits bid
    price: float
    size: float
    mid_at_trade: float


def _level_probabilities(L: int, decay: float) -> np.ndarray:
    w = np.exp(-decay * np.arange(L))
    w /= w.sum()
    return w


class LimitOrderBook:
    """
    Discrete-event LOB with L levels per side.
    External queue depth and MM queue depth are tracked per level.
    Market orders consume external depth first, then MM depth (MM joins the back of the queue).
    """

    def __init__(self, params: RegimeParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self.L = params.levels
        tick = params.tick_size
        base_tick = int(round(params.base_price / tick))

        self.best_bid_tick = base_tick - 1
        self.best_ask_tick = base_tick

        lvl = np.arange(self.L, dtype=float)
        mean_profile = params.base_depth * np.exp(-params.depth_decay * lvl)
        noise = self.rng.lognormal(mean=0.0, sigma=0.35, size=(2, self.L))

        self.bid_ext = mean_profile * noise[0]
        self.ask_ext = mean_profile * noise[1]
        self.bid_mm = np.zeros(self.L, dtype=float)
        self.ask_mm = np.zeros(self.L, dtype=float)

    def best_bid(self) -> float:
        return self.best_bid_tick * self.p.tick_size

    def best_ask(self) -> float:
        return self.best_ask_tick * self.p.tick_size

    def mid(self) -> float:
        return 0.5 * (self.best_bid() + self.best_ask())

    def spread(self) -> float:
        return self.best_ask() - self.best_bid()

    def top_depth(self, k: int = 1, include_mm: bool = True) -> Tuple[float, float]:
        k = min(k, self.L)
        if include_mm:
            bid = float(np.sum(self.bid_ext[:k] + self.bid_mm[:k]))
            ask = float(np.sum(self.ask_ext[:k] + self.ask_mm[:k]))
        else:
            bid = float(np.sum(self.bid_ext[:k]))
            ask = float(np.sum(self.ask_ext[:k]))
        return bid, ask

    def imbalance(self, k: int = 1) -> float:
        bid, ask = self.top_depth(k=k, include_mm=True)
        denom = bid + ask
        if denom <= 1e-12:
            return 0.0
        return (bid - ask) / denom

    def cancel_mm_all(self) -> None:
        self.bid_mm[:] = 0.0
        self.ask_mm[:] = 0.0

    def place_mm(self, side: Side, level_idx: int, size: float) -> None:
        if level_idx < 0 or level_idx >= self.L:
            return
        if side == Side.BID:
            self.bid_mm[level_idx] += size
        else:
            self.ask_mm[level_idx] += size

    def _refill_level(self) -> float:
        lvl = self.L - 1
        mean = self.p.base_depth * math.exp(-self.p.depth_decay * lvl)
        return float(mean * self.rng.lognormal(mean=0.0, sigma=0.45))

    def _shift_ask_up_one_tick(self) -> None:
        self.best_ask_tick += 1
        self.ask_ext[:-1] = self.ask_ext[1:]
        self.ask_mm[:-1] = self.ask_mm[1:]
        self.ask_ext[-1] = self._refill_level()
        self.ask_mm[-1] = 0.0

    def _shift_bid_down_one_tick(self) -> None:
        self.best_bid_tick -= 1
        self.bid_ext[:-1] = self.bid_ext[1:]
        self.bid_mm[:-1] = self.bid_mm[1:]
        self.bid_ext[-1] = self._refill_level()
        self.bid_mm[-1] = 0.0

    def _improve_bid_one_tick(self) -> bool:
        if self.best_bid_tick + 1 >= self.best_ask_tick:
            return False
        self.best_bid_tick += 1
        self.bid_ext[1:] = self.bid_ext[:-1]
        self.bid_mm[1:] = self.bid_mm[:-1]
        self.bid_ext[0] = 0.0
        self.bid_mm[0] = 0.0
        return True

    def _improve_ask_one_tick(self) -> bool:
        if self.best_ask_tick - 1 <= self.best_bid_tick:
            return False
        self.best_ask_tick -= 1
        self.ask_ext[1:] = self.ask_ext[:-1]
        self.ask_mm[1:] = self.ask_mm[:-1]
        self.ask_ext[0] = 0.0
        self.ask_mm[0] = 0.0
        return True

    def limit_arrival(self, side: Side, level_idx: int, size: float) -> None:
        size = max(0.0, float(size))
        if size <= 0:
            return

        if level_idx < 0:
            improve_ticks = -int(level_idx)
            for _ in range(improve_ticks):
                if side == Side.BID:
                    if not self._improve_bid_one_tick():
                        break
                else:
                    if not self._improve_ask_one_tick():
                        break
            if side == Side.BID:
                self.bid_ext[0] += size
            else:
                self.ask_ext[0] += size
            return

        if level_idx >= self.L:
            return
        if side == Side.BID:
            self.bid_ext[level_idx] += size
        else:
            self.ask_ext[level_idx] += size

    def cancel_arrival(self, side: Side, level_idx: int, frac: float) -> None:
        if level_idx < 0 or level_idx >= self.L:
            return
        frac = max(0.0, min(float(frac), 1.0))
        if side == Side.BID:
            self.bid_ext[level_idx] *= (1.0 - frac)
        else:
            self.ask_ext[level_idx] *= (1.0 - frac)

    def _best_ask_total(self) -> float:
        return float(self.ask_ext[0] + self.ask_mm[0])

    def _best_bid_total(self) -> float:
        return float(self.bid_ext[0] + self.bid_mm[0])

    def _repair_empty_best(self) -> None:
        for _ in range(self.L + 10):
            moved = False
            if self._best_ask_total() <= 1e-9:
                self._shift_ask_up_one_tick()
                moved = True
            if self._best_bid_total() <= 1e-9:
                self._shift_bid_down_one_tick()
                moved = True
            if self.best_ask_tick <= self.best_bid_tick:
                self.best_ask_tick = self.best_bid_tick + 1
                moved = True
            if not moved:
                break

    def market_order(self, q_initiator: int, size: float, event_idx: int, time_s: float) -> List[Trade]:
        size = max(0.0, float(size))
        fills: List[Trade] = []
        if size <= 0:
            return fills

        if q_initiator == +1:
            while size > 1e-9:
                self._repair_empty_best()
                px = self.best_ask()
                take_ext = min(self.ask_ext[0], size)
                self.ask_ext[0] -= take_ext
                size -= take_ext
                if size <= 1e-9:
                    break
                take_mm = min(self.ask_mm[0], size)
                if take_mm > 0:
                    fills.append(Trade(event_idx, time_s, +1, px, take_mm, self.mid()))
                    self.ask_mm[0] -= take_mm
                    size -= take_mm
                if self._best_ask_total() <= 1e-9:
                    self._shift_ask_up_one_tick()
        else:
            while size > 1e-9:
                self._repair_empty_best()
                px = self.best_bid()
                take_ext = min(self.bid_ext[0], size)
                self.bid_ext[0] -= take_ext
                size -= take_ext
                if size <= 1e-9:
                    break
                take_mm = min(self.bid_mm[0], size)
                if take_mm > 0:
                    fills.append(Trade(event_idx, time_s, -1, px, take_mm, self.mid()))
                    self.bid_mm[0] -= take_mm
                    size -= take_mm
                if self._best_bid_total() <= 1e-9:
                    self._shift_bid_down_one_tick()

        self._repair_empty_best()
        return fills


class BaseStrategy:
    def __init__(self, max_pos: float):
        self.pos = 0.0
        self.cash = 0.0
        self.max_pos = max_pos

    def mark_to_market(self, mid: float) -> float:
        return self.cash + self.pos * mid

    def on_fills(self, fills: List[Trade]) -> None:
        for tr in fills:
            if tr.q_initiator == +1:
                self.pos -= tr.size
                self.cash += tr.size * tr.price
            else:
                self.pos += tr.size
                self.cash -= tr.size * tr.price

    def update_quotes(self, book: LimitOrderBook, sigma2: float, t_s: float, day_end_s: float) -> Dict[str, float]:
        raise NotImplementedError


class AdaptiveMarketMaker(BaseStrategy):
    """
    Inventory-aware + imbalance-aware quoting.
    Reservation price + optimal spread follow Avellaneda-Stoikov-inspired formulas (in ticks),
    augmented by OFI drift and toxicity widening.
    """

    def __init__(self, params: StrategyParams):
        super().__init__(max_pos=params.max_pos)
        self.p = params

    def update_quotes(self, book: LimitOrderBook, sigma2_price_per_s: float, t_s: float, day_end_s: float) -> Dict[str, float]:
        book.cancel_mm_all()

        tick = book.p.tick_size
        mid = book.mid()
        mid_tick = mid / tick
        sigma2_tick = sigma2_price_per_s / (tick * tick + 1e-18)

        tau = min(self.p.terminal_horizon_s, max(0.0, day_end_s - t_s))
        gamma = self.p.gamma
        k = self.p.k

        ofi = book.imbalance(k=1)

        r = mid_tick - self.pos * gamma * sigma2_tick * tau
        r += self.p.ofi_drift_coeff * ofi

        spread = gamma * sigma2_tick * tau + (2.0 / gamma) * math.log(1.0 + gamma / k)
        spread += self.p.toxicity_spread_coeff * abs(ofi)

        bid_tick = math.floor(r - 0.5 * spread)
        ask_tick = math.ceil(r + 0.5 * spread)

        bid_tick = min(bid_tick, book.best_bid_tick)
        ask_tick = max(ask_tick, book.best_ask_tick)

        bid_level = int(book.best_bid_tick - bid_tick)
        ask_level = int(ask_tick - book.best_ask_tick)

        bid_level = max(0, min(bid_level, self.p.max_quote_levels_away, book.L - 1))
        ask_level = max(0, min(ask_level, self.p.max_quote_levels_away, book.L - 1))

        size = self.p.mm_order_size
        if self.pos < self.max_pos:
            book.place_mm(Side.BID, bid_level, size)
        if self.pos > -self.max_pos:
            book.place_mm(Side.ASK, ask_level, size)

        bid_tick_actual = book.best_bid_tick - bid_level
        ask_tick_actual = book.best_ask_tick + ask_level

        return {
            "mid": mid,
            "ofi": ofi,
            "sigma": math.sqrt(max(sigma2_price_per_s, 0.0)),
            "bid_level": float(bid_level),
            "ask_level": float(ask_level),
            "quote_spread": (ask_tick_actual - bid_tick_actual) * tick,
        }


class SymmetricMarketMaker(BaseStrategy):
    """
    Baseline: always quote at best bid/ask with fixed size (no inventory/OFI control).
    """

    def __init__(self, order_size: float, max_pos: float):
        super().__init__(max_pos=max_pos)
        self.order_size = order_size

    def update_quotes(self, book: LimitOrderBook, sigma2: float, t_s: float, day_end_s: float) -> Dict[str, float]:
        book.cancel_mm_all()
        size = self.order_size
        if self.pos < self.max_pos:
            book.place_mm(Side.BID, level_idx=0, size=size)
        if self.pos > -self.max_pos:
            book.place_mm(Side.ASK, level_idx=0, size=size)
        return {
            "mid": book.mid(),
            "ofi": book.imbalance(k=1),
            "sigma": math.sqrt(max(sigma2, 0.0)),
            "bid_level": 0.0,
            "ask_level": 0.0,
            "quote_spread": book.spread(),
        }


def compute_pnl_decomposition(trades: pd.DataFrame, mid_series: np.ndarray, horizon_events: int) -> Dict[str, float]:
    if len(trades) == 0:
        return {"spread_pnl": 0.0, "realized_spread_pnl": 0.0, "adverse_selection_pnl": 0.0}

    Q = trades["q_initiator"].to_numpy(dtype=float)
    px = trades["price"].to_numpy(dtype=float)
    sz = trades["size"].to_numpy(dtype=float)
    mid_t = trades["mid_at_trade"].to_numpy(dtype=float)

    spread_pnl = float(np.sum(Q * (px - mid_t) * sz))

    idx = trades["event_idx"].to_numpy(dtype=int)
    idx_f = np.minimum(idx + horizon_events, len(mid_series) - 1)
    mid_f = mid_series[idx_f]

    realized_spread_pnl = float(np.sum(Q * (px - mid_f) * sz))
    adverse_selection_pnl = float(np.sum(-Q * (mid_f - mid_t) * sz))
    return {
        "spread_pnl": spread_pnl,
        "realized_spread_pnl": realized_spread_pnl,
        "adverse_selection_pnl": adverse_selection_pnl,
    }


def simulate_day(
    regime: RegimeParams,
    strategy: BaseStrategy,
    n_events: int,
    seed: int,
    quote_interval_s: float,
    ewma_alpha: float = 0.05,
    realized_spread_horizon_events: int = 50,
    p_inside: float = 0.85,
    max_inside_improve: int = 10,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    book = LimitOrderBook(regime, rng)
    L = regime.levels
    lvl_probs = _level_probabilities(L, decay=regime.depth_decay)

    base_total = regime.lambda_limit + regime.lambda_cancel + regime.lambda_market
    target_rate = n_events / regime.day_seconds
    scale = target_rate / base_total
    lam_L = regime.lambda_limit * scale
    lam_C = regime.lambda_cancel * scale
    lam_M = regime.lambda_market * scale
    lam_total = lam_L + lam_C + lam_M

    t = 0.0
    day_end = regime.day_seconds
    next_quote = 0.0

    mid_prev = book.mid()
    sigma2 = (0.01 * mid_prev) ** 2 / regime.day_seconds

    times = np.empty(n_events, dtype=float)
    mids = np.empty(n_events, dtype=float)
    spreads = np.empty(n_events, dtype=float)
    inv = np.empty(n_events, dtype=float)
    pnl = np.empty(n_events, dtype=float)

    quote_log: List[Dict[str, float]] = []
    trades: List[Trade] = []

    quote_log.append(strategy.update_quotes(book, sigma2, t, day_end) | {"time_s": t, "event_idx": -1})
    next_quote += quote_interval_s

    for i in range(n_events):
        dt = rng.exponential(1.0 / lam_total)
        t += dt

        while next_quote <= t:
            info = strategy.update_quotes(book, sigma2, next_quote, day_end)
            quote_log.append(info | {"time_s": next_quote, "event_idx": i})
            next_quote += quote_interval_s

        u = rng.random() * lam_total
        if u < lam_L:
            et = EventType.LIMIT
        elif u < lam_L + lam_C:
            et = EventType.CANCEL
        else:
            et = EventType.MARKET

        if et == EventType.LIMIT:
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            spread_ticks = book.best_ask_tick - book.best_bid_tick
            if spread_ticks > 1 and rng.random() < p_inside:
                improve = int(rng.integers(1, min(max_inside_improve, spread_ticks - 1) + 1))
                level_idx = -improve
            else:
                level_idx = int(rng.choice(L, p=lvl_probs))
            size = float(rng.exponential(regime.limit_size_mean)) + 1.0
            book.limit_arrival(side, level_idx, size)
            book._repair_empty_best()

        elif et == EventType.CANCEL:
            bid_depth = float(np.sum(book.bid_ext))
            ask_depth = float(np.sum(book.ask_ext))
            side = Side.BID if rng.random() < (bid_depth / (bid_depth + ask_depth + 1e-12)) else Side.ASK
            weights = (book.bid_ext if side == Side.BID else book.ask_ext) + 1e-9
            weights = weights / weights.sum()
            level_idx = int(rng.choice(L, p=weights))
            frac = float(rng.uniform(0.0, regime.cancel_max_frac))
            book.cancel_arrival(side, level_idx, frac)
            book._repair_empty_best()

        else:
            imb = book.imbalance(k=1)
            p_buy = 0.5 + regime.ofi_sensitivity * imb
            p_buy = min(0.95, max(0.05, p_buy))
            q_initiator = +1 if rng.random() < p_buy else -1
            size = float(rng.exponential(regime.market_size_mean)) + 1.0
            if rng.random() < 0.12:
                size *= float(rng.integers(3, 10))
            fills = book.market_order(q_initiator, size, i, t)
            if fills:
                strategy.on_fills(fills)
                trades.extend(fills)

        mid_now = book.mid()
        inst_var_rate = ((mid_now - mid_prev) ** 2) / max(dt, 1e-9)
        sigma2 = (1.0 - ewma_alpha) * sigma2 + ewma_alpha * inst_var_rate
        mid_prev = mid_now

        times[i] = t
        mids[i] = mid_now
        spreads[i] = book.spread()
        inv[i] = strategy.pos
        pnl[i] = strategy.mark_to_market(mid_now)

    events = pd.DataFrame({
        "event_idx": np.arange(n_events),
        "time_s": times,
        "mid": mids,
        "spread": spreads,
        "inventory": inv,
        "pnl": pnl,
    })
    quotes = pd.DataFrame(quote_log).sort_values("time_s").reset_index(drop=True)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    inv_pnl = float(np.sum(inv[:-1] * np.diff(mids)))
    decomp = compute_pnl_decomposition(trades_df, mids, realized_spread_horizon_events)

    total_pnl = float(pnl[-1] - pnl[0])
    curve = pnl - pnl[0]
    peaks = np.maximum.accumulate(curve)
    max_dd = float(np.max(peaks - curve))

    capital = regime.base_price * strategy.max_pos
    ret = total_pnl / capital if capital > 0 else 0.0

    metrics = {
        "regime": regime.name,
        "strategy": strategy.__class__.__name__,
        "n_events": int(n_events),
        "sim_time_s": float(times[-1]),
        "trades": int(len(trades_df)),
        "final_inventory": float(inv[-1]),
        "total_pnl": total_pnl,
        "spread_pnl": decomp["spread_pnl"],
        "realized_spread_pnl": decomp["realized_spread_pnl"],
        "adverse_selection_pnl": decomp["adverse_selection_pnl"],
        "inventory_pnl": inv_pnl,
        "max_drawdown": max_dd,
        "return_on_capital": ret,
        "avg_spread": float(np.mean(spreads)),
        "avg_mid": float(np.mean(mids)),
    }

    return {"events": events, "quotes": quotes, "trades": trades_df, "metrics": metrics}


def simulate_many_days(
    regime: RegimeParams,
    strategy_factory,
    n_days: int,
    n_events: int,
    quote_interval_s: float,
    seed0: int = 100,
) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        strat = strategy_factory()
        out = simulate_day(regime, strat, n_events=n_events, seed=seed0 + d, quote_interval_s=quote_interval_s)
        rows.append(out["metrics"])
    return pd.DataFrame(rows)


def sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    return float(np.mean(r) / (np.std(r, ddof=1) + eps))


def plot_day(out: Dict[str, object], title: str, save_prefix: Optional[str] = None) -> None:
    events: pd.DataFrame = out["events"]
    quotes: pd.DataFrame = out["quotes"]

    t = events["time_s"].to_numpy()
    inv = events["inventory"].to_numpy()
    pnl = events["pnl"].to_numpy()
    spread = events["spread"].to_numpy()
    mid = events["mid"].to_numpy()

    # PnL component time series (event-index allocation)
    inv_pnl_series = np.cumsum(np.r_[0.0, inv[:-1] * np.diff(mid)])
    trades: pd.DataFrame = out["trades"]

    realized = np.zeros_like(mid)
    adverse = np.zeros_like(mid)
    if len(trades) > 0:
        Q = trades["q_initiator"].to_numpy(dtype=float)
        px = trades["price"].to_numpy(dtype=float)
        sz = trades["size"].to_numpy(dtype=float)
        mid_t = trades["mid_at_trade"].to_numpy(dtype=float)
        idx = trades["event_idx"].to_numpy(dtype=int)
        idx_f = np.minimum(idx + 50, len(mid) - 1)
        mid_f = mid[idx_f]
        rs = Q * (px - mid_f) * sz
        adv = -Q * (mid_f - mid_t) * sz
        np.add.at(realized, idx, rs)
        np.add.at(adverse, idx, adv)

    rs_cum = np.cumsum(realized)
    adv_cum = np.cumsum(adverse)
    total_cum = pnl - pnl[0]

    plt.figure(figsize=(10, 4))
    plt.plot(t, inv)
    plt.title(f"{title} — Inventory")
    plt.xlabel("Time (s)")
    plt.ylabel("Inventory (units)")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_inventory.png", dpi=180)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t, total_cum, label="Total PnL (MtM)")
    plt.plot(t, rs_cum, label="Realized spread")
    plt.plot(t, adv_cum, label="Adverse selection")
    plt.plot(t, inv_pnl_series, label="Inventory revaluation")
    plt.title(f"{title} — PnL Decomposition")
    plt.xlabel("Time (s)")
    plt.ylabel("PnL ($)")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_pnl_components.png", dpi=180)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t, spread, label="Market spread")
    plt.plot(quotes["time_s"].to_numpy(), quotes["quote_spread"].to_numpy(), label="MM quoted spread")
    plt.title(f"{title} — Spread Dynamics")
    plt.xlabel("Time (s)")
    plt.ylabel("Spread ($)")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_spreads.png", dpi=180)
    plt.show()


def main() -> None:
    low = RegimeParams(name="low_vol")
    high = RegimeParams(
        name="high_vol",
        base_depth=700.0,
        depth_decay=0.25,
        lambda_limit=1.0,
        lambda_cancel=0.9,
        lambda_market=0.8,
        market_size_mean=170.0,
        ofi_sensitivity=0.35,
    )
    params = StrategyParams()
    n_events = 50_000

    out_low = simulate_day(
        low, AdaptiveMarketMaker(params),
        n_events=n_events, seed=7, quote_interval_s=params.quote_interval_s
    )
    print("Low-vol day metrics:")
    print(pd.Series(out_low["metrics"]))
    plot_day(out_low, title="Adaptive MM (Low-vol)", save_prefix="lob_demo_low")

    out_high = simulate_day(
        high, AdaptiveMarketMaker(params),
        n_events=n_events, seed=8, quote_interval_s=params.quote_interval_s
    )
    print("\nHigh-vol day metrics:")
    print(pd.Series(out_high["metrics"]))

    n_days = 5
    df_low = simulate_many_days(
        low,
        strategy_factory=lambda: AdaptiveMarketMaker(params),
        n_days=n_days,
        n_events=n_events,
        quote_interval_s=params.quote_interval_s,
        seed0=1000,
    )
    df_high = simulate_many_days(
        high,
        strategy_factory=lambda: AdaptiveMarketMaker(params),
        n_days=n_days,
        n_events=n_events,
        quote_interval_s=params.quote_interval_s,
        seed0=2000,
    )
    print("\nRisk-adjusted summary (Adaptive MM):")
    for name, df in [("low_vol", df_low), ("high_vol", df_high)]:
        s = sharpe(df["return_on_capital"].to_numpy())
        print(f"{name}: mean return={df['return_on_capital'].mean(): .4f}, "
              f"std={df['return_on_capital'].std(ddof=1): .4f}, Sharpe={s: .2f}, "
              f"avg maxDD={df['max_drawdown'].mean(): .2f}")

    base_low = simulate_many_days(
        low,
        strategy_factory=lambda: SymmetricMarketMaker(order_size=params.mm_order_size, max_pos=params.max_pos),
        n_days=n_days,
        n_events=n_events,
        quote_interval_s=params.quote_interval_s,
        seed0=3000,
    )
    print("\nBaseline comparison (Low-vol):")
    print("Adaptive Sharpe:", sharpe(df_low["return_on_capital"].to_numpy()))
    print("Symmetric Sharpe:", sharpe(base_low["return_on_capital"].to_numpy()))


if __name__ == "__main__":
    main()
