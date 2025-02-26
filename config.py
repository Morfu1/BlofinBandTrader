import os
from dataclasses import dataclass, field
from typing import Literal, List, Optional
from dotenv import load_dotenv
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (only API keys)
env_path = Path('.env')
logger.info(f"Looking for .env file at: {env_path.absolute()}")
logger.info(f"File exists: {env_path.exists()}")

load_dotenv(dotenv_path=env_path)

# Debug: Print environment variables
logger.info(f"BLOFIN_API_KEY exists: {'BLOFIN_API_KEY' in os.environ}")
logger.info(f"BLOFIN_API_SECRET exists: {'BLOFIN_API_SECRET' in os.environ}")
logger.info(f"BLOFIN_PASSPHRASE exists: {'BLOFIN_PASSPHRASE' in os.environ}")


@dataclass
class Config:
    # API Configuration
    API_KEY: str = ""
    API_SECRET: str = ""
    API_PASSPHRASE: str = ""

    # Trading Pair Selection Configuration
    COIN_SELECTION_MODE: Literal[
        "single", "multiple",
        "top_volume"] = "multiple"  # Default selection mode
    SINGLE_COIN: str = "AR-USDT"  # Default trading pair for single mode
    MULTIPLE_COINS: List[str] = field(
        default_factory=lambda: ["AR-USDT", "NEAR-USDT", "BRETT-USDT", "POPCAT-USDT", "WIF-USDT", "TIA-USDT", "SOL-USDT", "OP-USDT", "XRP-USDT", "LINK-USDT", "ADA-USDT", "DOGE-USDT", "AVAX-USDT", "WLD-USDT", "PEPE-USDT", "INJ-USDT", "MKR-USDT", "JASMY-USDT", "PEOPLE-USDT", "TON-USDT", "KAS-USDT"])  # List of trading pairs
    TOP_VOLUME_COUNT: int = 10  # Number of top volume pairs to track

    # For backward compatibility
    TRADING_PAIR: str = "WIF-USDT"  # Will be set based on selection mode

    # Trading Parameters
    TIMEFRAME: Literal["1m", "3m", "5m", "15m", "30m", "1H", "4H", "1D"] = "1H"
    POSITION_SIZE: float = 100.0  # Margin size in USD
    LEVERAGE: int = 3
    POSITION_TYPE: Literal["isolated", "cross"] = "isolated"
    RISK_PER_TRADE: float = 0.01  # 1% risk per trade

    # Signal Parameters
    COMPLETE_CROSSOVER_THRESHOLD: float = 0.05
    SINGLE_MA_THRESHOLD: float = 0.05
    HIGH_VOLATILITY_THRESHOLD: float = 2.5

    # Risk Management Parameters
    TAKE_PROFIT_PERCENT: float = 2.0
    SL_PERCENTAGE: float = 1.0
    MAX_DRAWDOWN: float = 5.0
    TRAILING_STOP: float = 0.5
    RISK_REWARD_RATIO: float = 2.0
    STOP_LOSS_BUFFER: float = 0.5  # Buffer percentage to avoid stop hunts (0.5%)

    # Exchange Limits
    MIN_SIZE: float = 0.1
    MAX_SIZE: float = 1000.0
    CONTRACT_VALUE: float = 0.001

    # Strategy Parameters
    SMA_PERIOD: int = 21
    EMA_PERIOD: int = 34
    CANDLE_LOOKBACK: int = 10

    # API Endpoints
    REST_URL: str = "https://demo-trading-openapi.blofin.com"
    WS_PUBLIC_URL: str = "wss://demo-trading-openapi.blofin.com/ws/public"
    WS_PRIVATE_URL: str = "wss://demo-trading-openapi.blofin.com/ws/private"
    
    # Logging Configuration
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_CANDLE_UPDATES: bool = False  # Whether to log every candle update
    LOG_CONFIRMED_CANDLES_ONLY: bool = True  # Only log confirmed candles
    LOG_DETAILED_POSITION_UPDATES: bool = False  # Detailed position state logs
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        # Load API credentials from environment variables
        self.API_KEY = os.getenv("BLOFIN_API_KEY", "")
        self.API_SECRET = os.getenv("BLOFIN_API_SECRET", "")
        self.API_PASSPHRASE = os.getenv("BLOFIN_PASSPHRASE", "")

        # Validate required credentials
        if not all([self.API_KEY, self.API_SECRET, self.API_PASSPHRASE]):
            raise ValueError("API credentials not properly configured")

        # Set trading pair based on selection mode
        if self.COIN_SELECTION_MODE == "single":
            self.TRADING_PAIR = self.SINGLE_COIN
            self.MULTIPLE_COINS = [self.SINGLE_COIN]
        elif self.COIN_SELECTION_MODE == "multiple" and self.MULTIPLE_COINS:
            # Keep TRADING_PAIR for backward compatibility but don't rely on it
            self.TRADING_PAIR = self.MULTIPLE_COINS[0]
        elif self.COIN_SELECTION_MODE == "top_volume":
            # Will be populated later by MarketScanner
            self.MULTIPLE_COINS = []
            self.TRADING_PAIR = None
            
        # Configure logging based on settings
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format=self.LOG_FORMAT
        )

        # Log final configuration
        logger.info(
            f"\nFinal Trading Configuration:\n"
            f"========================================\n"
            f"Selection Mode: {self.COIN_SELECTION_MODE}\n"
            f"Active Coins: {self.MULTIPLE_COINS}\n"
            f"Timeframe: {self.TIMEFRAME}\n"
            f"Logging Level: {self.LOG_LEVEL}\n"
            f"Log Candle Updates: {self.LOG_CANDLE_UPDATES}\n"
            f"Log Confirmed Candles Only: {self.LOG_CONFIRMED_CANDLES_ONLY}\n"
            f"========================================")


# Create config instance
config = Config()
