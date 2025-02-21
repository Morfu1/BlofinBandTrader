from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from config import Config, config

class Indicators:
    def __init__(self, sma_period: int = 21, ema_period: int = 34, sl_percentage: float = 1.0, test_config: Optional[Config] = None):
        """Initialize technical indicators with configurable periods"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        if sma_period < 1 or ema_period < 1:
            raise ValueError("SMA and EMA periods must be positive integers")

        self.sma_period = sma_period
        self.ema_period = ema_period
        self.sl_percentage = sl_percentage

        # Use test configuration if provided, otherwise use global config
        self.config = test_config if test_config is not None else config

        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_symbol: Optional[str] = None

    def reset_state(self) -> None:
        """Reset the indicator state"""
        self.historical_data.clear()
        self.current_symbol = None
        self.logger.info("Indicator state reset")

    def initialize_historical_data(self, symbol: str, historical_prices: pd.Series) -> None:
        """Initialize historical data for a symbol"""
        try:
            if self.current_symbol is not None and symbol != self.current_symbol:
                self.logger.info(f"Switching from {self.current_symbol} to {symbol}")
                self.reset_state()

            if isinstance(historical_prices, pd.Series):
                if not isinstance(historical_prices.index, pd.DatetimeIndex):
                    historical_prices.index = pd.to_datetime(historical_prices.index)
                df = pd.DataFrame({'close': historical_prices})
            else:
                df = pd.DataFrame({'close': historical_prices})
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

            df = df.sort_index()
            self.historical_data[symbol] = df
            self.current_symbol = symbol

            self.logger.info(
                f"\nInitialized Historical Data for {symbol}:\n"
                f"Data points: {len(df)}\n"
                f"Latest price: {df['close'].iloc[-1]:.6f}")

            # Calculate and log initial MA values
            if len(df) >= max(self.sma_period, self.ema_period):
                sma, ema = self.calculate_bands(symbol)
                self.logger.debug(
                    f"Initial MA Values:\n"
                    f"SMA: {sma.iloc[-1]:.6f}\n"
                    f"EMA: {ema.iloc[-1]:.6f}")

        except Exception as e:
            self.logger.error(f"Error initializing historical data: {str(e)}")
            raise

    def update_real_time_data(self, symbol: str, timestamp: int, close_price: float) -> bool:
        """Update historical data with real-time market data"""
        try:
            self.validate_symbol(symbol)
            if not isinstance(timestamp, (int, float)) or not isinstance(close_price, (int, float)):
                raise ValueError("Invalid timestamp or close price format")

            dt = pd.to_datetime(timestamp, unit='ms')

            if len(self.historical_data[symbol]) > 0:
                last_timestamp = self.historical_data[symbol].index[-1]

                if dt == last_timestamp:
                    self.historical_data[symbol].loc[dt, 'close'] = close_price
                    self.logger.debug(f"Updated existing candle at {dt} with price {close_price:.6f}")
                else:
                    new_data = pd.DataFrame({'close': [close_price]}, index=[dt])
                    self.historical_data[symbol] = pd.concat([self.historical_data[symbol], new_data])
                    self.logger.debug(f"Added new candle at {dt} with price {close_price:.6f}")

                required_periods = max(self.sma_period, self.ema_period) + 10
                if len(self.historical_data[symbol]) > required_periods:
                    self.historical_data[symbol] = self.historical_data[symbol].tail(required_periods)

            return True

        except Exception as e:
            self.logger.error(f"Error updating real-time data: {str(e)}")
            return False

    def validate_symbol(self, symbol: str) -> None:
        """Validate symbol data exists"""
        if symbol != self.current_symbol:
            raise ValueError(f"Symbol mismatch: {symbol} vs {self.current_symbol}")
        if symbol not in self.historical_data:
            raise ValueError(f"No data for symbol {symbol}")

    @staticmethod
    def calculate_sma(data: pd.Series, period: int = 21) -> pd.Series:
        """Calculate Simple Moving Average"""
        return pd.to_numeric(data, errors='coerce').rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 34) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return pd.to_numeric(data, errors='coerce').ewm(span=period, adjust=False, min_periods=period).mean()

    def calculate_bands(self, symbol: str, current_close: Optional[float] = None) -> Tuple[pd.Series, pd.Series]:
        """Calculate SMA and EMA bands"""
        self.validate_symbol(symbol)
        df = self.historical_data[symbol].copy()

        if current_close is not None:
            if len(df) > 0:
                last_ts = df.index[-1]
                df.loc[last_ts, 'close'] = current_close

        if len(df) < max(self.sma_period, self.ema_period):
            self.logger.warning(f"Not enough data for {symbol}")
            return pd.Series(), pd.Series()

        sma = self.calculate_sma(df['close'], self.sma_period)
        ema = self.calculate_ema(df['close'], self.ema_period)

        return sma, ema

    def is_price_outside_bands(self, symbol: str, close: float, open_price: float) -> Tuple[bool, str]:
        """Check for signals based on candle open and close positions relative to bands"""
        try:
            sma, ema = self.calculate_bands(symbol, close)

            if sma.empty or ema.empty or len(sma) < 2 or len(ema) < 2:
                self.logger.warning("Not enough MA data points")
                return False, ""

            # Get current band values
            current_sma = sma.iloc[-1]
            current_ema = ema.iloc[-1]
            current_high = max(current_sma, current_ema)
            current_low = min(current_sma, current_ema)

            # Calculate percentage moves for signal validation
            price_move_pct = ((close - open_price) / open_price) * 100

            # Check for extreme volatility
            if len(self.historical_data[symbol]) >= 10:
                recent_volatility = self.historical_data[symbol]['close'].pct_change().tail(10).std() * 100
                if recent_volatility > self.config.HIGH_VOLATILITY_THRESHOLD:
                    self.logger.info(
                        f"\nSignal Rejected - High Volatility:\n"
                        f"Recent Volatility: {recent_volatility:.2f}%\n"
                        f"Threshold: {self.config.HIGH_VOLATILITY_THRESHOLD}%")
                    return False, ""

            # Enhanced signal analysis logging
            self.logger.debug(
                f"\nSignal Analysis Details:\n"
                f"{'-'*50}\n"
                f"Price Movement:\n"
                f"  Open Price: ${open_price:.6f}\n"
                f"  Close Price: ${close:.6f}\n"
                f"  Move: {price_move_pct:+.2f}%\n\n"
                f"Band Analysis:\n"
                f"  Upper Band: ${current_high:.6f}\n"
                f"  Lower Band: ${current_low:.6f}\n\n"
                f"Signal Thresholds:\n"
                f"  Crossover: {self.config.COMPLETE_CROSSOVER_THRESHOLD}%\n"
                f"  Single MA: {self.config.SINGLE_MA_THRESHOLD}%\n"
                f"  Volatility: {self.config.HIGH_VOLATILITY_THRESHOLD}%\n"
                f"{'-'*50}")

            # Scenario 1: Opening between bands and closing outside
            opening_between_bands = current_low <= open_price <= current_high

            if opening_between_bands:
                # Long signal: Closed above bands with sufficient movement
                if close > current_high and price_move_pct >= self.config.SINGLE_MA_THRESHOLD:
                    self.logger.info(
                        f"\n🔔 LONG Signal (Band Breakout):\n"
                        f"Opened between bands at ${open_price:.6f}\n"
                        f"Closed above bands at ${close:.6f}\n"
                        f"Move: {price_move_pct:+.2f}%")
                    return True, "long"

                # Short signal: Closed below bands with sufficient movement
                if close < current_low and price_move_pct <= -self.config.SINGLE_MA_THRESHOLD:
                    self.logger.info(
                        f"\n🔔 SHORT Signal (Band Breakout):\n"
                        f"Opened between bands at ${open_price:.6f}\n"
                        f"Closed below bands at ${close:.6f}\n"
                        f"Move: {price_move_pct:+.2f}%")
                    return True, "short"

            # Scenario 2: Complete band crossing
            # Opening below both bands and closing above
            if open_price < current_low and close > current_high:
                if price_move_pct >= self.config.COMPLETE_CROSSOVER_THRESHOLD:
                    self.logger.info(
                        f"\n🔔 LONG Signal (Complete Band Crossing):\n"
                        f"Opened below bands at ${open_price:.6f}\n"
                        f"Closed above bands at ${close:.6f}\n"
                        f"Move: {price_move_pct:+.2f}%")
                    return True, "long"

            # Opening above both bands and closing below
            if open_price > current_high and close < current_low:
                if price_move_pct <= -self.config.COMPLETE_CROSSOVER_THRESHOLD:
                    self.logger.info(
                        f"\n🔔 SHORT Signal (Complete Band Crossing):\n"
                        f"Opened above bands at ${open_price:.6f}\n"
                        f"Closed below bands at ${close:.6f}\n"
                        f"Move: {price_move_pct:+.2f}%")
                    return True, "short"

            self.logger.debug("No valid signal detected")
            return False, ""

        except Exception as e:
            self.logger.error(f"Error in is_price_outside_bands: {str(e)}")
            return False, ""

    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """Calculate stop loss based on percentage from entry price"""
        try:
            entry_price = float(entry_price)

            if position_type == "long":
                sl_price = entry_price * (1 - self.sl_percentage / 100)
            else:  # short position
                sl_price = entry_price * (1 + self.sl_percentage / 100)

            self.logger.info(
                f"\nStop Loss Calculation:\n"
                f"========================================\n"
                f"Position Type: {position_type}\n"
                f"Entry Price: ${entry_price:.6f}\n"
                f"Stop Loss Price: ${sl_price:.6f}\n"
                f"Stop Loss Percentage: {self.sl_percentage}%\n"
                f"Price Movement to SL: ${abs(sl_price - entry_price):.6f}\n"
                f"========================================")

            return sl_price

        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            raise