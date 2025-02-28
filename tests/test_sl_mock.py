import asyncio
import logging
import sys
import os
from decimal import Decimal

# Add the parent directory to the path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from band_based_sl import BandBasedStopLossManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("band_sl_test")

class MockRiskManager:
    """Mock risk manager for testing"""
    def __init__(self):
        self.logger = logging.getLogger("mock_risk_manager")
    
    def round_to_tick(self, price, position_type=None):
        """Mock method to round price to tick size"""
        return str(round(price, 6))
    
    async def get_latest_mark_price(self, trading_pair=None):
        """Mock method to get current price"""
        return 100.0  # Mock price
    
    async def get_current_positions(self, log_details=False):
        """Mock method to get current positions"""
        # Return a mock position
        return {
            'instId': 'BTC-USDT',
            'positions': '0.1',  # Long position
            'positionSide': 'long',
            'averagePrice': '95.0',
            'unrealizedPnl': '0.5',
            'slTriggerPrice': '90.0'
        }

class MockIndicators:
    """Mock indicators for testing"""
    def __init__(self):
        self.logger = logging.getLogger("mock_indicators")
    
    def calculate_bands(self, trading_pair, current_price=None):
        """Mock method to calculate bands"""
        import pandas as pd
        # Return mock SMA and EMA series
        return pd.Series([98.0]), pd.Series([97.0])

class TestBot:
    def __init__(self):
        """Initialize test bot with required components"""
        self.config = Config()
        self.risk_manager = MockRiskManager()
        self.indicators = MockIndicators()
        
        # Initialize band-based stop loss manager
        self.band_sl_manager = BandBasedStopLossManager(
            self.config,
            self.risk_manager,
            self.indicators
        )
        self.logger = logging.getLogger("band_sl_test")
    
    async def initialize(self):
        """Initialize the test bot"""
        self.logger.info("Initializing test bot...")
        self.logger.info("Test bot initialized")
        return True
    
    async def update_stop_loss(self, trading_pair, position_type, new_stop_loss):
        """Mock method to update stop loss"""
        self.logger.info(f"Would update SL for {trading_pair} to ${float(new_stop_loss)}")
        return True
    
    async def test_band_based_sl(self):
        """Test the band-based stop loss functionality"""
        self.logger.info("Starting band-based stop loss test")
        
        # Create a mock position
        trading_pair = "BTC-USDT"
        position_type = "long"
        entry_price = 95.0
        current_sl = 90.0
        
        # Initialize position tracking
        await self.band_sl_manager.initialize_position_tracking(
            trading_pair=trading_pair,
            position_side=position_type,
            entry_price=entry_price,
            initial_sl=current_sl
        )
        
        # Create a mock candle for testing
        from utils import Utils
        
        # Get current timestamp
        current_time = int(Decimal(str(int(asyncio.get_event_loop().time() * 1000))))
        
        # Mock current price higher than entry to simulate profit
        current_price = 105.0  # 10% profit
        
        mock_candle = {
            'ts': current_time,
            'open': current_price * 0.99,
            'high': current_price * 1.01,
            'low': current_price * 0.99,
            'close': current_price,
            'vol': 1000,
            'confirm': True
        }
        
        # Process the candle to test stop loss adjustment
        self.logger.info("Processing mock candle to test stop loss adjustment")
        
        # Get position
        position = await self.risk_manager.get_current_positions()
        
        # Update position state with correct keys
        self.band_sl_manager.position_states[trading_pair]['position_type'] = position_type
        
        # Process candle update
        new_sl = await self.band_sl_manager.process_candle_update(
            trading_pair, 
            mock_candle,
            position
        )
        
        if new_sl is not None:
            await self.update_stop_loss(trading_pair, position_type, new_sl)
        
        self.logger.info("Band-based stop loss test completed")

async def main():
    """Main function to run the test"""
    logger.info("Starting band-based stop loss test")
    
    test_bot = TestBot()
    await test_bot.initialize()
    await test_bot.test_band_based_sl()
    
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main())
