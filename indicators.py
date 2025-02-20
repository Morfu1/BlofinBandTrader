from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime

class Indicators:
    def __init__(self, sma_period: int = 21, ema_period: int = 34, sl_percentage: float = 1.0):
        """
        Initialize technical indicators with configurable periods

        Args:
            sma_period: Period for Simple Moving Average calculation
            ema_period: Period for Exponential Moving Average calculation
            sl_percentage: Stop loss percentage from entry price
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Validate periods
        if sma_period < 1 or ema_period < 1:
            raise ValueError("SMA and EMA periods must be positive integers")

        self.sma_period = sma_period
        self.ema_period = ema_period
        self.sl_percentage = sl_percentage

        # Store historical data for each symbol with timestamp as index
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_symbol: Optional[str] = None  # Fixed type hint

    def reset_state(self) -> None:
        """Reset the indicator state, clearing all historical data"""
        self.historical_data.clear()
        self.current_symbol = None
        self.logger.info("Indicator state reset")

    def initialize_historical_data(self, symbol: str, historical_prices: pd.Series) -> None:
        """
        Initialize or update historical data for a specific trading symbol

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            historical_prices: Series of historical closing prices for the symbol
        """
        try:
            # If switching symbols, clear existing data
            if self.current_symbol is not None and symbol != self.current_symbol:
                self.logger.info(f"Switching from {self.current_symbol} to {symbol}, clearing historical data")
                self.reset_state()

            # Ensure we have a DataFrame with datetime index
            if isinstance(historical_prices, pd.Series):
                if not isinstance(historical_prices.index, pd.DatetimeIndex):
                    self.logger.warning("Converting index to DatetimeIndex")
                    historical_prices.index = pd.to_datetime(historical_prices.index)
                df = pd.DataFrame({'close': historical_prices})
            else:
                df = pd.DataFrame({'close': historical_prices})
                if not isinstance(df.index, pd.DatetimeIndex):
                    self.logger.warning("Converting index to DatetimeIndex")
                    df.index = pd.to_datetime(df.index)

            # Sort by timestamp to ensure chronological order
            df = df.sort_index()

            # Store with proper datetime index
            self.historical_data[symbol] = df
            self.current_symbol = symbol

            # Format dates for logging
            start_date = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
            end_date = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')

            self.logger.info(
                f"\nInitialized Historical Data for {symbol}:\n"
                f"========================================\n"
                f"Data points: {len(df)}\n"
                f"Date range: {start_date} to {end_date}\n"
                f"Latest price: {df['close'].iloc[-1]:.6f}\n"
                f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB\n"
                f"========================================")

        except Exception as e:
            self.logger.error(f"Error initializing historical data: {str(e)}")
            raise

    def update_real_time_data(self, symbol: str, timestamp: int, close_price: float) -> bool:
        """
        Update historical data with real-time market data

        Args:
            symbol: Trading symbol
            timestamp: Unix timestamp in milliseconds
            close_price: Latest closing price

        Returns:
            bool: True if update was successful
        """
        try:
            # Validate symbol and inputs
            self.validate_symbol(symbol)
            if not isinstance(timestamp, (int, float)) or not isinstance(close_price, (int, float)):
                raise ValueError("Invalid timestamp or close price format")

            # Convert timestamp to datetime
            dt = pd.to_datetime(timestamp, unit='ms')

            # Update existing candle or append new one
            if len(self.historical_data[symbol]) > 0:
                last_timestamp = self.historical_data[symbol].index[-1]

                if dt == last_timestamp:
                    # Update existing candle
                    self.historical_data[symbol].loc[dt, 'close'] = close_price
                    self.logger.debug(f"Updated existing candle at {dt} with price {close_price:.6f}")
                else:
                    # Append new candle
                    new_data = pd.DataFrame({'close': [close_price]}, index=[dt])
                    self.historical_data[symbol] = pd.concat([self.historical_data[symbol], new_data])
                    self.logger.debug(f"Added new candle at {dt} with price {close_price:.6f}")

                # Keep only required number of candles
                required_periods = max(self.sma_period, self.ema_period) + 10  # Extra buffer
                if len(self.historical_data[symbol]) > required_periods:
                    self.historical_data[symbol] = self.historical_data[symbol].tail(required_periods)

            return True

        except Exception as e:
            self.logger.error(f"Error updating real-time data: {str(e)}")
            return False

    def validate_symbol(self, symbol: str) -> None:
        """Validate that we're using data from the correct symbol"""
        if symbol != self.current_symbol:
            raise ValueError(
                f"Symbol mismatch: Attempting to use data for {symbol} "
                f"but initialized with {self.current_symbol}")

        if symbol not in self.historical_data:
            raise ValueError(
                f"No historical data initialized for symbol {symbol}. "
                "Call initialize_historical_data first.")

    @staticmethod
    def calculate_sma(data: pd.Series, period: int = 21) -> pd.Series:
        """Calculate Simple Moving Average"""
        data = pd.to_numeric(data, errors='coerce')
        return data.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 34) -> pd.Series:
        """Calculate Exponential Moving Average"""
        data = pd.to_numeric(data, errors='coerce')
        return data.ewm(span=period, adjust=False, min_periods=period).mean()

    def calculate_bands(self, symbol: str, current_close: Optional[float] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SMA and EMA bands using stored period values and symbol-specific data

        Args:
            symbol: Trading symbol to calculate bands for
            current_close: Optional current closing price to append to historical data

        Returns:
            Tuple of (SMA, EMA) Series
        """
        # Validate symbol and data
        self.validate_symbol(symbol)
        df = self.historical_data[symbol].copy()

        # Update with current close if provided
        if current_close is not None:
            if len(df) > 0:
                df.loc[df.index[-1], 'close'] = current_close

        # Ensure we have enough data
        if len(df) < max(self.sma_period, self.ema_period):
            self.logger.warning(
                f"Not enough data for {symbol}. Need at least "
                f"{max(self.sma_period, self.ema_period)} candles")
            return pd.Series(), pd.Series()

        self.logger.debug(
            f"\nBand Calculation for {symbol}:\n"
            f"Data points: {len(df)}\n"
            f"Latest price: {df['close'].iloc[-1]:.6f}\n"
            f"Latest timestamp: {df.index[-1]}")

        sma = self.calculate_sma(df['close'], self.sma_period)
        ema = self.calculate_ema(df['close'], self.ema_period)

        return sma, ema

    def is_price_outside_bands(self, symbol: str, close: float) -> Tuple[bool, str]:
        """
        Check if price has properly crossed and closed outside the bands

        Args:
            symbol: Trading symbol to check
            close: Current closing price

        Returns:
            Tuple(is_valid_signal: bool, position_type: str)
        """
        sma, ema = self.calculate_bands(symbol, close)

        if sma.empty or ema.empty or len(sma) < 2 or len(ema) < 2:
            self.logger.warning("Not enough data for signal detection")
            return False, ""

        # Get current and previous values
        current_sma = sma.iloc[-1]
        current_ema = ema.iloc[-1]
        prev_sma = sma.iloc[-2]
        prev_ema = ema.iloc[-2]

        # Calculate band levels
        current_high = max(current_sma, current_ema)
        current_low = min(current_sma, current_ema)
        prev_high = max(prev_sma, prev_ema)
        prev_low = min(prev_sma, prev_ema)

        # Get historical data for the symbol
        df = self.historical_data[symbol]
        if len(df) < 10:  # Need enough data to check volatility
            return False, ""

        # Check for flat MAs (no trend)
        sma_change = abs((current_sma - prev_sma) / prev_sma) * 100
        ema_change = abs((current_ema - prev_ema) / prev_ema) * 100

        if sma_change < 0.01 and ema_change < 0.01:
            return False, ""

        # Check for extreme volatility (using last 10 candles)
        recent_volatility = df['close'].pct_change().tail(10).std() * 100
        if recent_volatility > 2.0:  # More than 2% standard deviation
            return False, ""

        # Get previous close for crossover detection
        prev_close = df['close'].iloc[-2]

        # Complete Crossover Scenarios
        if close > current_high and prev_close <= prev_high:
            self.logger.info(
                f"\nSignal Analysis:\n"
                f"========================================\n"
                f"Type: BULLISH CROSSOVER\n"
                f"Previous Close: ${prev_close:.6f}\n"
                f"Current Close: ${close:.6f}\n"
                f"Band High: ${current_high:.6f}\n"
                f"Price Movement: {((close/prev_close)-1)*100:.2f}%\n"
                f"Volatility: {recent_volatility:.2f}%\n"
                f"SMA Change: {sma_change:.4f}%\n"
                f"EMA Change: {ema_change:.4f}%\n"
                f"========================================")
            return True, "long"

        elif close < current_low and prev_close >= prev_low:
            self.logger.info(
                f"\nSignal Analysis:\n"
                f"========================================\n"
                f"Type: BEARISH CROSSOVER\n"
                f"Previous Close: ${prev_close:.6f}\n"
                f"Current Close: ${close:.6f}\n"
                f"Band Low: ${current_low:.6f}\n"
                f"Price Movement: {((close/prev_close)-1)*100:.2f}%\n"
                f"Volatility: {recent_volatility:.2f}%\n"
                f"SMA Change: {sma_change:.4f}%\n"
                f"EMA Change: {ema_change:.4f}%\n"
                f"========================================")
            return True, "short"

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