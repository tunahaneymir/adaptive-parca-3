"""
Adaptive Parameters - Parça 3
Based on: pa-strateji2 Parça 3

Adaptive Parameter System:
- ATR-based volatility adjustment
- Timeframe-based scaling
- Dynamic parameter calculation for:
  * ZigZag depth/deviation
  * Swing strength
  * (Future: EMA periods, thresholds - Parça 8)
- Real-time adaptation to market conditions
"""

from __future__ import annotations
from typing import Dict, Optional, Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class AdaptiveParams:
    """Adaptive parameters result"""
    # Original base values
    base_zigzag_depth: int
    base_zigzag_deviation: float
    base_swing_strength: int
    
    # Adapted values
    adapted_zigzag_depth: int
    adapted_zigzag_deviation: float
    adapted_swing_strength: int
    
    # Multipliers used
    atr_multiplier: float
    timeframe_multiplier: float
    
    # Context
    atr_percent: float
    volatility_regime: str  # "LOW", "NORMAL", "HIGH", "EXTREME"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'zigzag_depth': self.adapted_zigzag_depth,
            'zigzag_deviation': self.adapted_zigzag_deviation,
            'swing_strength': self.adapted_swing_strength,
            'atr_percent': self.atr_percent,
            'volatility_regime': self.volatility_regime,
            'atr_mult': self.atr_multiplier,
            'tf_mult': self.timeframe_multiplier
        }


class AdaptiveParameterCalculator:
    """
    Adaptive Parameter Calculator
    
    Calculates dynamic parameters based on:
    1. Market Volatility (ATR%)
    2. Timeframe
    3. (Future: Coin characteristics, historical performance)
    
    Volatility Regimes:
    - EXTREME: ATR > 8% → Mult 1.5x (Very volatile, wider params)
    - HIGH: ATR 5-8% → Mult 1.2x (Above average volatility)
    - NORMAL: ATR 3-5% → Mult 1.0x (Normal market)
    - LOW: ATR < 3% → Mult 0.8x (Low volatility, tighter params)
    
    Timeframe Scaling:
    - 4H: Mult 1.5x (Longer view, wider params)
    - 1H: Mult 1.0x (Standard)
    - 15M: Mult 0.7x (Shorter view, tighter params)
    
    Usage:
        calc = AdaptiveParameterCalculator(config)
        
        params = calc.calculate(
            high=high,
            low=low,
            close=close,
            timeframe="1H"
        )
        
        # Use adapted parameters
        zigzag_depth = params.adapted_zigzag_depth
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Adaptive Parameter Calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get base parameters from config
        zone_config = config.get('zones', {}) if config else {}
        zigzag_config = zone_config.get('zigzag', {})
        swing_config = zone_config.get('swing', {})
        
        # Base parameters (from config or defaults)
        self.base_zigzag_depth = zigzag_config.get('depth', 12)
        self.base_zigzag_deviation = zigzag_config.get('deviation', 5)
        self.base_swing_strength = swing_config.get('strength', 5)
        
        # Volatility thresholds (ATR%)
        self.extreme_volatility = 8.0  # >8% = extreme
        self.high_volatility = 5.0     # 5-8% = high
        self.normal_volatility = 3.0   # 3-5% = normal
        # <3% = low
        
        # Multipliers
        self.extreme_mult = 1.5
        self.high_mult = 1.2
        self.normal_mult = 1.0
        self.low_mult = 0.8
        
        # Timeframe multipliers
        self.timeframe_mults = {
            '4H': 1.5,
            '1H': 1.0,
            '15M': 0.7
        }
        
        # Limits (prevent extreme values)
        self.min_zigzag_depth = 5
        self.max_zigzag_depth = 30
        self.min_zigzag_deviation = 2
        self.max_zigzag_deviation = 15
        self.min_swing_strength = 3
        self.max_swing_strength = 15
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: Optional[np.ndarray] = None,
        timeframe: str = "1H"
    ) -> AdaptiveParams:
        """
        Calculate adaptive parameters
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            atr: Optional ATR values (will calculate if not provided)
            timeframe: Timeframe ("4H", "1H", "15M")
            
        Returns:
            AdaptiveParams with adapted values
        """
        # Calculate ATR if not provided
        if atr is None:
            atr = self._calculate_atr(high, low, close)
        
        current_price = close[-1]
        current_atr = atr[-1]
        
        # Calculate ATR as percentage of price
        atr_percent = (current_atr / current_price) * 100
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Determine ATR Multiplier
        # ═══════════════════════════════════════════════════════════
        atr_mult, volatility_regime = self._get_atr_multiplier(atr_percent)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Get Timeframe Multiplier
        # ═══════════════════════════════════════════════════════════
        tf_mult = self.timeframe_mults.get(timeframe, 1.0)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Calculate Adapted Parameters
        # ═══════════════════════════════════════════════════════════
        
        # ZigZag Depth
        adapted_depth = int(self.base_zigzag_depth * atr_mult * tf_mult)
        adapted_depth = self._clamp(
            adapted_depth,
            self.min_zigzag_depth,
            self.max_zigzag_depth
        )
        
        # ZigZag Deviation
        adapted_deviation = int(self.base_zigzag_deviation * atr_mult)
        adapted_deviation = self._clamp(
            adapted_deviation,
            self.min_zigzag_deviation,
            self.max_zigzag_deviation
        )
        
        # Swing Strength
        adapted_swing = int(self.base_swing_strength * atr_mult)
        adapted_swing = self._clamp(
            adapted_swing,
            self.min_swing_strength,
            self.max_swing_strength
        )
        
        return AdaptiveParams(
            base_zigzag_depth=self.base_zigzag_depth,
            base_zigzag_deviation=self.base_zigzag_deviation,
            base_swing_strength=self.base_swing_strength,
            adapted_zigzag_depth=adapted_depth,
            adapted_zigzag_deviation=adapted_deviation,
            adapted_swing_strength=adapted_swing,
            atr_multiplier=atr_mult,
            timeframe_multiplier=tf_mult,
            atr_percent=atr_percent,
            volatility_regime=volatility_regime
        )
    
    def _get_atr_multiplier(self, atr_percent: float) -> tuple[float, str]:
        """
        Get ATR-based multiplier and volatility regime
        
        Args:
            atr_percent: ATR as percentage of price
            
        Returns:
            (multiplier, regime_name)
        """
        if atr_percent > self.extreme_volatility:
            return self.extreme_mult, "EXTREME"
        elif atr_percent > self.high_volatility:
            return self.high_mult, "HIGH"
        elif atr_percent > self.normal_volatility:
            return self.normal_mult, "NORMAL"
        else:
            return self.low_mult, "LOW"
    
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate ATR"""
        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # ATR = EMA of TR
        ema = np.zeros_like(tr, dtype=float)
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            ema[i] = (tr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value between min and max"""
        return max(min_val, min(value, max_val))
    
    def get_regime_description(self, regime: str) -> str:
        """Get human-readable regime description"""
        descriptions = {
            "EXTREME": "Extreme volatility - Very wide parameters",
            "HIGH": "High volatility - Wider parameters",
            "NORMAL": "Normal volatility - Standard parameters",
            "LOW": "Low volatility - Tighter parameters"
        }
