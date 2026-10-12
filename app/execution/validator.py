"""Symbol-aware order validator for pre-execution safety.

Validates order requests against broker symbol specifications:
- Volume limits (min, max, step)
- SL/TP direction (BUY: SL < entry < TP, SELL: TP < entry < SL)
- Minimum stop distance (stops level)
- Tick size / digits alignment
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from .base import OrderRequest

log = get_logger(__name__)


@dataclass
class SymbolSpecification:
    symbol: str = "XAUUSD"
    digits: int = 2
    tick_size: float = 0.01
    tick_value: float = 1.0
    contract_size: float = 100.0
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    min_stop_distance_points: float = 0.50  # In price points (e.g. $0.50 on Gold)


DEFAULT_SYMBOL_SPECS: dict[str, SymbolSpecification] = {
    "XAUUSD": SymbolSpecification(
        symbol="XAUUSD",
        digits=2,
        tick_size=0.01,
        tick_value=1.0,
        contract_size=100.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        min_stop_distance_points=0.50,
    ),
    "EURUSD": SymbolSpecification(
        symbol="EURUSD",
        digits=5,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100000.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        min_stop_distance_points=0.00030,
    ),
    "GBPUSD": SymbolSpecification(
        symbol="GBPUSD",
        digits=5,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100000.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        min_stop_distance_points=0.00030,
    ),
    "BTCUSD": SymbolSpecification(
        symbol="BTCUSD",
        digits=2,
        tick_size=0.01,
        tick_value=1.0,
        contract_size=1.0,
        min_volume=0.01,
        max_volume=50.0,
        volume_step=0.01,
        min_stop_distance_points=10.0,
    ),
}


@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    spec: SymbolSpecification | None = None


class OrderValidator:
    """Pre-execution order validator ensuring strict broker compliance."""

    def __init__(self, specs: dict[str, SymbolSpecification] | None = None) -> None:
        self.specs = specs or DEFAULT_SYMBOL_SPECS

    def get_spec(self, symbol: str) -> SymbolSpecification:
        normalized = symbol.upper().strip()
        if normalized in self.specs:
            return self.specs[normalized]
        # Fallback default specification for unknown symbols
        return SymbolSpecification(symbol=normalized)

    def validate(
        self,
        request: OrderRequest,
        spec: SymbolSpecification | None = None,
        current_bid: float | None = None,
        current_ask: float | None = None,
    ) -> ValidationResult:
        """Validate an OrderRequest against symbol specifications and market rules."""
        sym_spec = spec or self.get_spec(request.symbol)
        side = request.side.lower().strip()

        # 1. Side validity
        if side not in ("buy", "sell"):
            err = f"Invalid order side: {request.side!r} (must be 'buy' or 'sell')"
            log.warning("Order validation failed: %s", err)
            return ValidationResult(valid=False, error=err, spec=sym_spec)

        # 2. Volume checks
        vol = request.volume
        if vol <= 0 or math.isnan(vol) or math.isinf(vol):
            err = f"Invalid order volume: {vol}"
            log.warning("Order validation failed: %s", err)
            return ValidationResult(valid=False, error=err, spec=sym_spec)

        if vol < sym_spec.min_volume:
            err = f"Volume {vol:.4f} is below minimum {sym_spec.min_volume:.4f} for {sym_spec.symbol}"
            log.warning("Order validation failed: %s", err)
            return ValidationResult(valid=False, error=err, spec=sym_spec)

        if vol > sym_spec.max_volume:
            err = f"Volume {vol:.4f} exceeds maximum {sym_spec.max_volume:.4f} for {sym_spec.symbol}"
            log.warning("Order validation failed: %s", err)
            return ValidationResult(valid=False, error=err, spec=sym_spec)

        # Volume step alignment check
        step = sym_spec.volume_step
        if step > 0:
            remainder = (vol - sym_spec.min_volume) % step
            # Account for floating point inaccuracy
            if abs(remainder) > 1e-5 and abs(remainder - step) > 1e-5:
                err = f"Volume {vol:.4f} does not align with volume step {step:.4f} for {sym_spec.symbol}"
                log.warning("Order validation failed: %s", err)
                return ValidationResult(valid=False, error=err, spec=sym_spec)

        # 3. Entry price validity
        entry = request.entry_ref_price
        if entry <= 0 or math.isnan(entry) or math.isinf(entry):
            err = f"Invalid entry reference price: {entry}"
            log.warning("Order validation failed: %s", err)
            return ValidationResult(valid=False, error=err, spec=sym_spec)

        # 4. SL / TP Direction & Minimum Stop Distance
        sl = request.stop_loss
        tp = request.take_profit
        min_stop_dist = sym_spec.min_stop_distance_points

        if side == "buy":
            # For BUY: SL < entry < TP
            if sl is not None:
                if sl >= entry:
                    err = f"Invalid BUY stop loss: SL ({sl:.5f}) must be strictly below entry price ({entry:.5f})"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)
                if (entry - sl) < min_stop_dist - 1e-6:
                    err = f"BUY stop loss distance ({entry - sl:.5f}) is below minimum stop distance ({min_stop_dist:.5f}) for {sym_spec.symbol}"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)

            if tp is not None:
                if tp <= entry:
                    err = f"Invalid BUY take profit: TP ({tp:.5f}) must be strictly above entry price ({entry:.5f})"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)
                if (tp - entry) < min_stop_dist - 1e-6:
                    err = f"BUY take profit distance ({tp - entry:.5f}) is below minimum stop distance ({min_stop_dist:.5f}) for {sym_spec.symbol}"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)

        elif side == "sell":
            # For SELL: TP < entry < SL
            if sl is not None:
                if sl <= entry:
                    err = f"Invalid SELL stop loss: SL ({sl:.5f}) must be strictly above entry price ({entry:.5f})"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)
                if (sl - entry) < min_stop_dist - 1e-6:
                    err = f"SELL stop loss distance ({sl - entry:.5f}) is below minimum stop distance ({min_stop_dist:.5f}) for {sym_spec.symbol}"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)

            if tp is not None:
                if tp >= entry:
                    err = f"Invalid SELL take profit: TP ({tp:.5f}) must be strictly below entry price ({entry:.5f})"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)
                if (entry - tp) < min_stop_dist - 1e-6:
                    err = f"SELL take profit distance ({entry - tp:.5f}) is below minimum stop distance ({min_stop_dist:.5f}) for {sym_spec.symbol}"
                    log.warning("Order validation failed: %s", err)
                    return ValidationResult(valid=False, error=err, spec=sym_spec)

        return ValidationResult(valid=True, error="", spec=sym_spec)