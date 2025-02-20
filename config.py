import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in attached_assets directory
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

    # Trading Parameters
    TRADING_PAIR: str = "AR-USDT"
    TIMEFRAME: Literal["1m", "3m", "5m", "15m", "30m", "1H", "4H",
                       "1D"] = "5m"  # Default to 1m
    POSITION_SIZE: float = 100.0  # Margin size in USD
    LEVERAGE: int = 3
    POSITION_TYPE: Literal["isolated", "cross"] = "isolated"
    RISK_PER_TRADE: float = 0.01  # 1% risk per trade

    # Exchange Limits (from Blofin API docs)
    MIN_SIZE: float = 0.1  # Minimum order size in contracts
    MAX_SIZE: float = 1000.0  # Maximum position size in contracts
    CONTRACT_VALUE: float = 0.001  # Value of each contract in base currency

    # Strategy Parameters
    SMA_PERIOD: int = 21
    EMA_PERIOD: int = 34
    TAKE_PROFIT_PERCENT: float = 2.0  # Default 2%
    CANDLE_LOOKBACK: int = 10  # For SL calculation

    # API Endpoints
    REST_URL: str = "https://demo-trading-openapi.blofin.com"
    WS_PUBLIC_URL: str = "wss://demo-trading-openapi.blofin.com/ws/public"
    WS_PRIVATE_URL: str = "wss://demo-trading-openapi.blofin.com/ws/private"

    def __post_init__(self):
        # Load API credentials from environment variables
        self.API_KEY = os.getenv("BLOFIN_API_KEY", "")
        self.API_SECRET = os.getenv("BLOFIN_API_SECRET", "")
        self.API_PASSPHRASE = os.getenv("BLOFIN_PASSPHRASE", "")

        # Log loaded values (without exposing sensitive data)
        logger.info(f"API_KEY loaded: {bool(self.API_KEY)}")
        logger.info(f"API_SECRET loaded: {bool(self.API_SECRET)}")
        logger.info(f"API_PASSPHRASE loaded: {bool(self.API_PASSPHRASE)}")
        logger.info(f"Using trading pair: {self.TRADING_PAIR}")

        # Load trading parameters from environment variables with validation
        env_timeframe = os.getenv("TIMEFRAME")
        logger.info(
            f"Environment TIMEFRAME value: {env_timeframe or 'not set, using default: 1m'}"
        )

        if env_timeframe:
            if env_timeframe in [
                    "1m", "3m", "5m", "15m", "30m", "1H", "4H", "1D"
            ]:
                self.TIMEFRAME = env_timeframe
                logger.info(
                    f"TIMEFRAME set from environment: {self.TIMEFRAME}")
            else:
                logger.warning(
                    f"Invalid TIMEFRAME in environment: {env_timeframe}, using default: {self.TIMEFRAME}"
                )

        if os.getenv("POSITION_SIZE"):
            try:
                self.POSITION_SIZE = float(os.getenv("POSITION_SIZE"))
                logger.info(
                    f"POSITION_SIZE set from environment: {self.POSITION_SIZE}"
                )
            except ValueError:
                logger.warning(
                    f"Invalid POSITION_SIZE in environment, using default: {self.POSITION_SIZE}"
                )

        if os.getenv("LEVERAGE"):
            try:
                self.LEVERAGE = int(os.getenv("LEVERAGE"))
                logger.info(f"LEVERAGE set from environment: {self.LEVERAGE}")
            except ValueError:
                logger.warning(
                    f"Invalid LEVERAGE in environment, using default: {self.LEVERAGE}"
                )

        if os.getenv("POSITION_TYPE"):
            position_type = os.getenv("POSITION_TYPE").lower()
            if position_type in ["isolated", "cross"]:
                self.POSITION_TYPE = position_type
                logger.info(
                    f"POSITION_TYPE set from environment: {self.POSITION_TYPE}"
                )
            else:
                logger.warning(
                    f"Invalid POSITION_TYPE in environment, using default: {self.POSITION_TYPE}"
                )

        if not all([self.API_KEY, self.API_SECRET, self.API_PASSPHRASE]):
            raise ValueError("API credentials not properly configured")

        # Log final configuration
        logger.info(
            f"Final configuration - TIMEFRAME: {self.TIMEFRAME}, TRADING_PAIR: {self.TRADING_PAIR}"
        )


# Create config instance
config = Config()
