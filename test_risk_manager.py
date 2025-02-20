import unittest
import asyncio
from decimal import Decimal
from risk_manager import RiskManager
from config import Config
from datetime import datetime, timedelta

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.risk_manager = RiskManager(config=self.config)

        # Initialize with test values
        self.risk_manager.contract_value = Decimal('1000.00000000')
        self.risk_manager.min_size = Decimal('1.00000000')
        self.risk_manager.lot_size = Decimal('1.00000000')
        self.risk_manager.max_leverage = Decimal('50')
        self.risk_manager.max_market_size = Decimal('3000.00000000')
        self.risk_manager.tick_size = Decimal('0.00000100')

    def test_position_size_calculation(self):
        """Test position size calculation with USDT margin"""
        current_price = 0.002911
        position_info = self.risk_manager.calculate_position_size(current_price)

        self.assertIsNotNone(position_info)
        self.assertEqual(position_info['size'], '103.00000000')  # Match exact format
        self.assertEqual(position_info['leverage'], '3')
        self.assertEqual(position_info['marginMode'], 'isolated')
        self.assertAlmostEqual(float(position_info['actualMargin']), 99.94, places=2)

    def test_round_to_tick(self):
        """Test price rounding to tick size"""
        test_cases = [
            (0.002911, '0.00291100'),  # Match exact format from risk_manager
            (0.0029116, '0.00291100'),  # Round down
            (0.002911001, '0.00291100')  # Round to tick
        ]

        for price, expected in test_cases:
            rounded = self.risk_manager.round_to_tick(price)
            self.assertEqual(rounded, expected)

    def test_stop_loss_calculation(self):
        """Test stop loss calculation and rounding"""
        async def run_test():
            entry_price = 0.002911

            # Mock historical candle data
            mock_candles = [
                {
                    'timestamp': datetime.now().timestamp(),
                    'high': 0.002940,
                    'low': 0.002881
                }
                for _ in range(10)
            ]

            # Mock the get_historical_candles method
            async def mock_get_historical_candles():
                return mock_candles

            self.risk_manager.get_historical_candles = mock_get_historical_candles

            # Test long position
            sl_long = await self.risk_manager.calculate_stop_loss(entry_price, "long")
            self.assertEqual(sl_long, 0.00288100)  # Lowest wick rounded to tick size

            # Test short position
            sl_short = await self.risk_manager.calculate_stop_loss(entry_price, "short")
            self.assertEqual(sl_short, 0.00294000)  # Highest wick rounded to tick size

        # Run the async test
        asyncio.run(run_test())

    def test_take_profit_calculation(self):
        """Test take profit calculation and rounding"""
        entry_price = 0.002911

        # First calculate stop loss values
        sl_long = 0.002890  # Use predefined values for testing
        sl_short = 0.002940

        # Test long position
        tp_long = self.risk_manager.calculate_take_profit(entry_price, sl_long, "long")
        risk_distance = abs(entry_price - sl_long)
        self.assertEqual(tp_long, float(self.risk_manager.round_to_tick(entry_price + (2 * risk_distance))))

        # Test short position
        tp_short = self.risk_manager.calculate_take_profit(entry_price, sl_short, "short")
        risk_distance = abs(entry_price - sl_short)
        self.assertEqual(tp_short, float(self.risk_manager.round_to_tick(entry_price - (2 * risk_distance))))

if __name__ == '__main__':
    unittest.main()