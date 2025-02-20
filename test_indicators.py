import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from indicators import Indicators

class TestIndicators(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.indicators = Indicators(sma_period=21, ema_period=34, sl_percentage=1.0)
        self.symbol = "TEST-USDT"
        
    def generate_test_data(self, prices: list, start_time: datetime = None) -> pd.Series:
        """Generate test price data with timestamps"""
        if start_time is None:
            start_time = datetime.now()
        
        dates = [start_time + timedelta(minutes=5*i) for i in range(len(prices))]
        return pd.Series(prices, index=dates)

    def test_complete_bullish_crossover(self):
        """Test complete bullish crossover detection"""
        # Generate prices that will create a complete bullish crossover
        # Price moves from below both MAs to above both MAs
        prices = [10.0] * 40  # Initial flat prices
        prices.extend([10.0 + 0.1*i for i in range(10)])  # Gradual increase
        prices[-2] = 11.8  # Previous candle below bands
        prices[-1] = 12.2  # Current candle above bands
        
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)
        
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1]  # Current close price
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(position_type, "long")

    def test_complete_bearish_crossover(self):
        """Test complete bearish crossover detection"""
        # Generate prices for bearish crossover
        prices = [10.0] * 40
        prices.extend([10.0 - 0.1*i for i in range(10)])
        prices[-2] = 8.2  # Previous candle above bands
        prices[-1] = 7.8  # Current candle below bands
        
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)
        
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1]
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(position_type, "short")

    def test_single_ma_bullish_crossover(self):
        """Test single MA bullish crossover from between bands"""
        # Price starts between bands, then crosses above upper band
        prices = [10.0] * 40
        prices[-3] = 10.2  # Between bands
        prices[-2] = 10.3  # Between bands
        prices[-1] = 10.8  # Above bands
        
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)
        
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1]
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(position_type, "long")

    def test_invalid_flat_mas(self):
        """Test rejection of signals when MAs are flat"""
        # Generate completely flat prices
        prices = [10.0] * 50
        
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)
        
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            10.5  # Slightly above flat MAs
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_invalid_high_volatility(self):
        """Test rejection during high volatility"""
        # Generate highly volatile prices
        prices = [10.0] * 40
        prices.extend([10.0 + ((-1)**i)*i*0.5 for i in range(10)])  # Add volatile swings
        
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)
        
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1]
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_price_touching_not_closing(self):
        """Test rejection when price only touches but doesn't close beyond bands"""
        prices = [10.0] * 40
        prices[-2] = 10.0  # Previous close at midpoint
        prices[-1] = 10.1  # Current close just touching upper band
        
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)
        
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1]
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

if __name__ == '__main__':
    unittest.main()
