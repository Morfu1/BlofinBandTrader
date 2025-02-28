"""Test suite for the Indicators class"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from indicators import Indicators
from config import Config
import logging

class TestConfig(Config):
    """Test configuration with appropriate thresholds for test data"""
    def __init__(self):
        super().__init__()
        # Use actual config values for tests
        self.TRADING_PAIR = "TEST-USDT"
        self.COMPLETE_CROSSOVER_THRESHOLD = 0.1  # 0.1% threshold for complete crossovers
        self.SINGLE_MA_THRESHOLD = 0.1  # 0.1% threshold for single MA breakouts
        self.HIGH_VOLATILITY_THRESHOLD = 2.5  # 2.5% threshold for volatility

class TestIndicators(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Create test configuration
        self.test_config = TestConfig()

        # Initialize indicators with test configuration
        self.indicators = Indicators(
            sma_period=21,
            ema_period=34,
            sl_percentage=1.0,
            test_config=self.test_config
        )
        self.symbol = "TEST-USDT"

    def generate_test_data(self, prices: list, start_time: datetime = None) -> pd.Series:
        """Generate test price data with timestamps"""
        if start_time is None:
            start_time = datetime.now()

        dates = [start_time + timedelta(minutes=5*i) for i in range(len(prices))]
        return pd.Series(prices, index=dates)

    def test_between_bands_breakout_long(self):
        """Test signal generation for opening between bands and closing above"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Create gentle trend to establish bands
        prices.extend([10.0 + 0.0005*i for i in range(15)])  # Reduced slope

        # Add final candle with proper threshold move
        open_price = 10.005  # Opening between bands
        close_price = 10.016  # Closing above bands (+0.11% move, exceeding threshold)

        prices[-1] = open_price
        prices.append(close_price)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Verify signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            close_price,
            open_price
        )

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "long")

    def test_between_bands_breakout_short(self):
        """Test signal generation for opening between bands and closing below"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Create gentle trend to establish bands
        prices.extend([10.0 - 0.0005*i for i in range(15)])  # Reduced slope

        # Add final candle with proper threshold move
        open_price = 9.995  # Opening between bands
        close_price = 9.984  # Closing below bands (-0.11% move, exceeding threshold)

        prices[-1] = open_price
        prices.append(close_price)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Verify signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            close_price,
            open_price
        )

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "short")

    def test_complete_band_crossing_long(self):
        """Test signal generation for opening below bands and closing above"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Create steeper trend to establish clear bands
        prices.extend([10.0 + 0.001*i for i in range(15)])

        # Add final candle with complete band crossing
        open_price = 9.99  # Opening below bands
        close_price = 10.012  # Closing above bands (+0.22% move, exceeding threshold)

        prices[-1] = open_price
        prices.append(close_price)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Verify signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            close_price,
            open_price
        )

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "long")

    def test_complete_band_crossing_short(self):
        """Test signal generation for opening above bands and closing below"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Create steeper trend to establish clear bands
        prices.extend([10.0 - 0.001*i for i in range(15)])
        # Add final candle with complete band crossing
        open_price = 10.01  # Opening above bands
        close_price = 9.988  # Closing below bands (-0.22% move, exceeding threshold)

        prices[-1] = open_price
        prices.append(close_price)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Verify signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            close_price,
            open_price
        )

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "short")

    def test_complete_bullish_crossover(self):
        """Test complete bullish crossover detection"""
        # Generate baseline data with bands ABOVE price initially
        prices = [10.0] * 21  # Initial flat prices for SMA calculation
        # Create downtrend to establish bands above price
        prices.extend([10.0 - 0.02*i for i in range(15)])  # Steeper downtrend
        prices.extend([9.7] * 5)  # Consolidation below bands
        # Create clear bullish crossover with gap above bands
        prices[-2] = 9.7  # Previous close below bands
        prices[-1] = 9.981  # Current close above bands (+2.89% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Get band values for logging
        sma, ema = self.indicators.calculate_bands(self.symbol, prices[-1])
        current_high = max(sma.iloc[-1], ema.iloc[-1])
        current_low = min(sma.iloc[-1], ema.iloc[-1])
        prev_high = max(sma.iloc[-2], ema.iloc[-2])
        prev_low = min(sma.iloc[-2], ema.iloc[-2])

        # Detailed logging before signal check
        self.logger.debug(f"\nTest Details - Complete Bullish Crossover:")
        self.logger.debug(f"Previous close: ${prices[-2]:.6f}")
        self.logger.debug(f"Current close: ${prices[-1]:.6f}")
        self.logger.debug(f"Price movement: {((prices[-1] - prices[-2]) / prices[-2]) * 100:.2f}%")
        self.logger.debug(f"Band Boundaries:")
        self.logger.debug(f"  Current: ${current_low:.6f} - ${current_high:.6f}")
        self.logger.debug(f"  Previous: ${prev_low:.6f} - ${prev_high:.6f}")
        self.logger.debug(f"Thresholds: Complete={self.test_config.COMPLETE_CROSSOVER_THRESHOLD}%, Single={self.test_config.SINGLE_MA_THRESHOLD}%")

        # Check signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.logger.debug(f"Signal valid: {is_valid}")
        self.logger.debug(f"Position type: {position_type}")

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "long")

    def test_complete_bearish_crossover(self):
        """Test complete bearish crossover detection"""
        # Generate baseline data with bands BELOW price initially
        prices = [10.0] * 21  # Initial flat prices for SMA calculation
        prices.extend([10.0 + 0.02*i for i in range(15)])  # Steeper uptrend
        prices.extend([10.3] * 5)  # Consolidation above bands
        # Create clear bearish crossover with gap below bands
        prices[-2] = 10.3  # Previous close above bands
        prices[-1] = 10.019  # Current close below bands (-2.73% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Get band values for logging
        sma, ema = self.indicators.calculate_bands(self.symbol, prices[-1])
        current_high = max(sma.iloc[-1], ema.iloc[-1])
        current_low = min(sma.iloc[-1], ema.iloc[-1])
        prev_high = max(sma.iloc[-2], ema.iloc[-2])
        prev_low = min(sma.iloc[-2], ema.iloc[-2])

        # Detailed logging before signal check
        self.logger.debug(f"\nTest Details - Complete Bearish Crossover:")
        self.logger.debug(f"Previous close: ${prices[-2]:.6f}")
        self.logger.debug(f"Current close: ${prices[-1]:.6f}")
        self.logger.debug(f"Price movement: {((prices[-1] - prices[-2]) / prices[-2]) * 100:.2f}%")
        self.logger.debug(f"Band Boundaries:")
        self.logger.debug(f"  Current: ${current_low:.6f} - ${current_high:.6f}")
        self.logger.debug(f"  Previous: ${prev_low:.6f} - ${prev_high:.6f}")
        self.logger.debug(f"Thresholds: Complete={self.test_config.COMPLETE_CROSSOVER_THRESHOLD}%, Single={self.test_config.SINGLE_MA_THRESHOLD}%")

        # Check signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.logger.debug(f"Signal valid: {is_valid}")
        self.logger.debug(f"Position type: {position_type}")

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "short")

    def test_invalid_flat_mas(self):
        """Test rejection of signals when MAs are flat"""
        # Generate completely flat prices
        prices = [10.0] * 40
        # Add tiny movement that shouldn't trigger
        prices[-2] = 10.0  # Previous close at flat level
        prices[-1] = 10.01  # Current close slightly above (0.1% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_invalid_high_volatility(self):
        """Test rejection during high volatility"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Add extreme volatile swings
        for i in range(10):
            mult = 0.3 * (i + 1)  # Increasing multiplier
            if i % 2 == 0:
                prices.append(10.0 + mult)  # Strong up move
            else:
                prices.append(10.0 - mult)  # Strong down move

        # Add crossover move that should be rejected due to volatility
        prices[-2] = 9.5  # Previous close
        prices[-1] = 9.8  # Current close (+3.16% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_price_touching_not_closing(self):
        """Test rejection when price only touches but doesn't close beyond bands"""
        prices = [10.0] * 30
        # Create very gentle uptrend to establish bands
        prices.extend([10.0 + 0.005*i for i in range(20)])
        prices[-2] = 10.1  # Previous close near midpoint
        prices[-1] = 10.12  # Current close just touching band (0.2% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_single_ma_upward_breakout(self):
        """Test upward breakout from between bands"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Create very gentle uptrend to establish stable bands
        prices.extend([10.0 + 0.0001*i for i in range(15)])  # Minimal slope
        prices.extend([10.001] * 5)  # Initial consolidation at lower level

        # Initialize with preliminary data
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Get band values to position the previous close
        sma, ema = self.indicators.calculate_bands(self.symbol)
        band_high = max(sma.iloc[-1], ema.iloc[-1])
        band_low = min(sma.iloc[-1], ema.iloc[-1])

        # Set previous close to midpoint of bands
        prev_close = band_low + (band_high - band_low) / 2
        self.logger.debug(f"Setting previous close to band midpoint: ${prev_close:.6f}")

        # Update test data with positioned previous close and breakout
        prices[-1] = prev_close  # Set previous close between bands
        prices.append(prev_close * 1.0025)  # Breakout move (+0.25%)

        # Reinitialize with final data
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Get final band values for verification
        sma, ema = self.indicators.calculate_bands(self.symbol, prices[-2])  # Get previous bands
        prev_high = max(sma.iloc[-1], ema.iloc[-1])
        prev_low = min(sma.iloc[-1], ema.iloc[-1])

        # Enhanced logging
        self.logger.debug(f"\nTest Details - Single MA Upward Breakout:")
        self.logger.debug(f"Previous close: ${prices[-2]:.6f}")
        self.logger.debug(f"Current close: ${prices[-1]:.6f}")
        self.logger.debug(f"Price movement: {((prices[-1] - prices[-2]) / prices[-2]) * 100:.2f}%")
        self.logger.debug(f"Previous Band Boundaries: ${prev_low:.6f} - ${prev_high:.6f}")
        self.logger.debug(f"Band spread: {((prev_high - prev_low) / prev_low) * 100:.4f}%")

        # Verify previous close is between bands with tolerance
        tolerance = 0.000001  # Small floating point tolerance
        self.assertTrue(
            prev_low - tolerance <= prices[-2] <= prev_high + tolerance,
            f"Previous close (${prices[-2]:.6f}) not between bands (${prev_low:.6f} - ${prev_high:.6f})"
        )

        # Check signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2] #passing previous close as open price for consistency.
        )

        self.logger.debug(f"Signal valid: {is_valid}")
        self.logger.debug(f"Position type: {position_type}")

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "long")

    def test_single_ma_downward_breakout(self):
        """Test downward breakout from between bands"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Create very gentle downtrend to establish stable bands
        prices.extend([10.0 - 0.0001*i for i in range(15)])  # Minimal slope
        prices.extend([9.999] * 5)  # Initial consolidation at higher level

        # Initialize with preliminary data
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Get band values to position the previous close
        sma, ema = self.indicators.calculate_bands(self.symbol)
        band_high = max(sma.iloc[-1], ema.iloc[-1])
        band_low = min(sma.iloc[-1], ema.iloc[-1])

        # Set previous close to midpoint of bands
        prev_close = band_low + (band_high - band_low) / 2
        self.logger.debug(f"Setting previous close to band midpoint: ${prev_close:.6f}")

        # Update test data with positioned previous close and breakout
        prices[-1] = prev_close  # Set previous close between bands
        prices.append(prev_close * 0.9975)  # Breakout move (-0.25%)

        # Reinitialize with final data
        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        # Get final band values for verification
        sma, ema = self.indicators.calculate_bands(self.symbol, prices[-2])  # Get previous bands
        prev_high = max(sma.iloc[-1], ema.iloc[-1])
        prev_low = min(sma.iloc[-1], ema.iloc[-1])

        # Enhanced logging
        self.logger.debug(f"\nTest Details - Single MA Downward Breakout:")
        self.logger.debug(f"Previous close: ${prices[-2]:.6f}")
        self.logger.debug(f"Current close: ${prices[-1]:.6f}")
        self.logger.debug(f"Price movement: {((prices[-1] - prices[-2]) / prices[-2]) * 100:.2f}%")
        self.logger.debug(f"Previous Band Boundaries: ${prev_low:.6f} - ${prev_high:.6f}")
        self.logger.debug(f"Band spread: {((prev_high - prev_low) / prev_low) * 100:.4f}%")

        # Verify previous close is between bands with tolerance
        tolerance = 0.000001  # Small floating point tolerance
        self.assertTrue(
            prev_low - tolerance <= prices[-2] <= prev_high + tolerance,
            f"Previous close (${prices[-2]:.6f}) not between bands (${prev_low:.6f} - ${prev_high:.6f})"
        )

        # Check signal
        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2] #passing previous close as open price for consistency.
        )

        self.logger.debug(f"Signal valid: {is_valid}")
        self.logger.debug(f"Position type: {position_type}")

        self.assertTrue(is_valid)
        self.assertEqual(position_type, "short")

    def test_invalid_flat_mas(self):
        """Test rejection of signals when MAs are flat"""
        # Generate completely flat prices
        prices = [10.0] * 40
        # Add tiny movement that shouldn't trigger
        prices[-2] = 10.0  # Previous close at flat level
        prices[-1] = 10.01  # Current close slightly above (0.1% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_invalid_high_volatility(self):
        """Test rejection during high volatility"""
        # Generate baseline data
        prices = [10.0] * 21  # Initial flat prices
        # Add extreme volatile swings
        for i in range(10):
            mult = 0.3 * (i + 1)  # Increasing multiplier
            if i % 2 == 0:
                prices.append(10.0 + mult)  # Strong up move
            else:
                prices.append(10.0 - mult)  # Strong down move

        # Add crossover move that should be rejected due to volatility
        prices[-2] = 9.5  # Previous close
        prices[-1] = 9.8  # Current close (+3.16% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

    def test_price_touching_not_closing(self):
        """Test rejection when price only touches but doesn't close beyond bands"""
        prices = [10.0] * 30
        # Create very gentle uptrend to establish bands
        prices.extend([10.0 + 0.005*i for i in range(20)])
        prices[-2] = 10.1  # Previous close near midpoint
        prices[-1] = 10.12  # Current close just touching band (0.2% move)

        historical_prices = self.generate_test_data(prices)
        self.indicators.initialize_historical_data(self.symbol, historical_prices)

        is_valid, position_type = self.indicators.is_price_outside_bands(
            self.symbol,
            prices[-1],
            prices[-2]  # Using previous close as open price
        )

        self.assertFalse(is_valid)
        self.assertEqual(position_type, "")

if __name__ == '__main__':
    unittest.main()