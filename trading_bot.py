from typing import Dict, List, Optional
import asyncio
import pandas as pd
import logging
import aiohttp
from datetime import datetime, timedelta
from indicators import Indicators
from config import Config
from utils import Utils
from websocket_manager import WebSocketManager
from risk_manager import RiskManager  # Using the root risk_manager.py


class TradingBot:

    def __init__(self):
        """Initialize bot components and configuration"""
        self.config = Config()
        self.indicators = Indicators(sma_period=self.config.SMA_PERIOD,
                                     ema_period=self.config.EMA_PERIOD,
                                     sl_percentage=1.0)
        self.risk_manager = RiskManager(config=self.config)
        self.websocket_manager: Optional[WebSocketManager] = None
        self.candles_df = pd.DataFrame()
        self.position_open = False
        self.logger = logging.getLogger(__name__)

        # Track position direction
        self.long_position_open = False
        self.short_position_open = False

        # Always use trading pair from config
        self.current_trading_pair = self.config.TRADING_PAIR
        self.logger.info(
            f"Initialized with trading pair: {self.current_trading_pair} on {self.config.TIMEFRAME} timeframe"
        )

    async def handle_position_updates(self, position_data: Dict):
        """Handle position updates and state management"""
        try:
            # Check if position exists
            if not position_data:
                # Reset all position flags if no positions exist
                self.position_open = False
                self.long_position_open = False
                self.short_position_open = False
                self.logger.info("All position flags reset - no active positions")
                return

            # Determine position direction from size
            position_size = float(position_data.get('positions', 0))

            # Update position flags based on position size
            if position_size > 0:  # Long position
                self.long_position_open = True
                self.short_position_open = False
            elif position_size < 0:  # Short position
                self.long_position_open = False
                self.short_position_open = True
            else:  # No position
                self.long_position_open = False
                self.short_position_open = False

            # Update overall position state
            self.position_open = self.long_position_open or self.short_position_open

            self.logger.info(
                f"\nPosition State Updated:\n"
                f"========================================\n"
                f"Position Size: {position_size}\n"
                f"Long Position: {self.long_position_open}\n"
                f"Short Position: {self.short_position_open}\n"
                f"Any Position Open: {self.position_open}\n"
                f"========================================")

        except Exception as e:
            self.logger.error(f"Error handling position update: {str(e)}")
            self.logger.exception("Full traceback:")

    async def initialize(self):
        """Initialize the trading bot with clean state"""
        Utils.setup_logging()
        self.logger.info(
            f"Initializing trading bot for {self.current_trading_pair} on {self.config.TIMEFRAME} timeframe..."
        )

        # Reset state
        self.position_open = False
        self.long_position_open = False
        self.short_position_open = False
        self.candles_df = pd.DataFrame()

        # Always ensure we're using the current config trading pair
        self.current_trading_pair = self.config.TRADING_PAIR  # Refresh from config

        # Initialize RiskManager with current trading pair parameters
        await self.risk_manager.initialize()

        # Check for any existing positions and update state
        current_position = await self.risk_manager.get_current_positions()
        await self.handle_position_updates(current_position)

        # Reset indicators state
        self.indicators.reset_state()

        # Load fresh historical data
        await self.load_historical_data()

        # Setup WebSocket with current trading pair
        self.websocket_manager = WebSocketManager(self.config)
        await self.websocket_manager.connect()

        # Register candle update callback
        self.websocket_manager.register_callback('candle',
                                                 self.handle_candle_update)
        self.logger.info(
            f"Registered candle update callback for {self.current_trading_pair} on {self.config.TIMEFRAME} timeframe"
        )

        # Initialize indicators with clean data
        if not self.candles_df.empty:
            self.indicators.initialize_historical_data(
                self.current_trading_pair, self.candles_df['close'])

            # Safe logging with DataFrame checks
            min_price = 0 if self.candles_df.empty else float(
                self.candles_df['close'].min())
            max_price = 0 if self.candles_df.empty else float(
                self.candles_df['close'].max())

            self.logger.info(
                f"\nInitialization Complete:\n"
                f"========================================\n"
                f"Trading Pair: {self.current_trading_pair}\n"
                f"Timeframe: {self.config.TIMEFRAME}\n"
                f"Candles Count: {len(self.candles_df)}\n"
                f"Current Price Range: ${min_price:.6f} - ${max_price:.6f}\n"
                f"Memory Usage: {self.candles_df.memory_usage().sum() / 1024:.2f} KB\n"
                f"Position State: Long={self.long_position_open}, Short={self.short_position_open}\n"
                f"WebSocket Status: Connected\n"
                f"Callback Registration: Complete\n"
                f"========================================")
        else:
            self.logger.warning(
                f"\nInitialization Warning:\n"
                f"========================================\n"
                f"No historical data available for {self.current_trading_pair}\n"
                f"Please check API connectivity and trading pair configuration\n"
                f"========================================")

    async def load_historical_data(self):
        """Load initial historical candlestick data"""
        url = f"{self.config.REST_URL}/api/v1/market/candles"
        params = {
            "instId": self.current_trading_pair,
            "bar": self.config.TIMEFRAME,  # Use timeframe from config
            "limit": "100"
        }

        self.logger.info(
            f"Loading historical data for {self.current_trading_pair} on {self.config.TIMEFRAME} timeframe"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data['code'] == '0':
                    # Clear existing data to prevent mixing with old coin data
                    self.candles_df = pd.DataFrame()

                    # Convert data to DataFrame
                    candles = data['data']
                    df = pd.DataFrame(candles,
                                      columns=[
                                          'ts', 'open', 'high', 'low', 'close',
                                          'vol', 'volCurrency',
                                          'volCurrencyQuote', 'confirm'
                                      ])

                    # Add trading pair column for data isolation
                    df['trading_pair'] = self.current_trading_pair

                    # Ensure all numeric columns are properly converted
                    df = df.astype({
                        'ts': float,
                        'open': float,
                        'high': float,
                        'low': float,
                        'close': float,
                        'vol': float,
                        'volCurrency': float,
                        'volCurrencyQuote': float,
                        'confirm': int,
                        'trading_pair': str
                    })

                    # Convert timestamps to datetime and set as index
                    df.index = pd.to_datetime(df['ts'].astype(int), unit='ms')
                    df = df.sort_index()  # Ensure chronological order

                    # Store the processed DataFrame
                    self.candles_df = df

                    # Initialize historical data for indicators with sorted data
                    self.indicators.initialize_historical_data(
                        self.current_trading_pair, self.candles_df['close'])

                    # Format dates for logging
                    start_date = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
                    end_date = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')

                    self.logger.info(
                        f"\nHistorical Data Loaded:\n"
                        f"========================================\n"
                        f"Trading Pair: {self.current_trading_pair}\n"
                        f"Timeframe: {self.config.TIMEFRAME}\n"
                        f"Number of candles: {len(df)}\n"
                        f"Price range: ${df['close'].min():.6f} - ${df['close'].max():.6f}\n"
                        f"Time range: {start_date} - {end_date}\n"
                        f"DataFrame Memory Usage: {self.candles_df.memory_usage().sum() / 1024:.2f} KB\n"
                        f"========================================")
                else:
                    raise Exception(
                        f"Failed to load historical data: {data['msg']}")

    async def handle_candle_update(self, candle_data: List):
        """Handle real-time candlestick updates"""
        try:
            # Convert timestamp to human-readable format for logging
            timestamp_ms = float(candle_data[0][0])
            datetime_str = datetime.fromtimestamp(
                timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')

            # Extract and validate candle data with enhanced logging
            try:
                candle = {
                    'ts':
                    float(candle_data[0][0]),
                    'open':
                    float(candle_data[0][1]),
                    'high':
                    float(candle_data[0][2]),
                    'low':
                    float(candle_data[0][3]),
                    'close':
                    float(candle_data[0][4]),
                    'vol':
                    float(candle_data[0][5]),
                    'volCurrency':
                    float(candle_data[0][6]),
                    'volCurrencyQuote':
                    float(candle_data[0][7]),
                    'confirm':
                    bool(int(float(candle_data[0][8])))
                    if len(candle_data[0]) > 8 else False,
                }
                self.logger.info(
                    f"\nReceived new candle data at {datetime_str}:\n"
                    f"Open: ${candle['open']:.6f}\n"
                    f"High: ${candle['high']:.6f}\n"
                    f"Low: ${candle['low']:.6f}\n"
                    f"Close: ${candle['close']:.6f}\n"
                    f"Volume: {candle['vol']:.2f}\n"
                    f"Confirmed: {candle['confirm']}")
            except (IndexError, ValueError) as e:
                self.logger.error(f"Invalid candle data format: {str(e)}")
                return

            # Calculate indicators with debug logging
            sma, ema = self.indicators.calculate_bands(
                self.current_trading_pair, candle['close'])
            self.logger.info(f"\nIndicator calculations:\n"
                             f"SMA Period: {self.indicators.sma_period}\n"
                             f"EMA Period: {self.indicators.ema_period}\n"
                             f"Latest SMA: ${sma.iloc[-1]:.6f}\n"
                             f"Latest EMA: ${ema.iloc[-1]:.6f}")

            # Get latest values if we have enough data
            if not sma.empty and not ema.empty:
                current_sma = sma.iloc[-1]
                current_ema = ema.iloc[-1]
                band_high = max(current_sma, current_ema)
                band_low = min(current_sma, current_ema)

                # Calculate price positions for current candle
                current_price_above = candle['close'] > band_high
                current_price_below = candle['close'] < band_low
                current_price_between = band_low <= candle['close'] <= band_high

                self.logger.info(
                    f"\nPrice position analysis:\n"
                    f"Current close: ${candle['close']:.6f}\n"
                    f"Band high: ${band_high:.6f}\n"
                    f"Band low: ${band_low:.6f}\n"
                    f"Price above bands: {current_price_above}\n"
                    f"Price below bands: {current_price_below}\n"
                    f"Price between bands: {current_price_between}")

                # Check for potential signals with enhanced logging
                if not self.position_open and candle['confirm']:
                    is_valid_signal, signal_type = self.indicators.is_price_outside_bands(
                        self.current_trading_pair, candle['close'])

                    self.logger.info(
                        f"\nSignal detection:\n"
                        f"Position open: {self.position_open}\n"
                        f"Candle confirmed: {candle['confirm']}\n"
                        f"Valid signal detected: {is_valid_signal}\n"
                        f"Signal type: {signal_type if signal_type else 'None'}"
                    )

            # Check for signals only if candle is confirmed
            if candle['confirm']:
                await self.check_signals(pd.Series(candle))

        except Exception as e:
            self.logger.error(f"Error handling candle update: {str(e)}")
            self.logger.exception("Full traceback:")

    async def check_signals(self, current_candle: pd.Series):
        """Check for trading signals"""
        try:
            # Filter candles for current trading pair BEFORE any calculations
            pair_candles = self.candles_df[self.candles_df['trading_pair'] ==
                                           self.current_trading_pair].copy()

            # Only proceed if we have enough data and candle is confirmed
            if len(pair_candles) < 2 or not current_candle['confirm']:
                return

            # Calculate indicators with the current candle's close price
            current_close = float(current_candle['close'])
            sma, ema = self.indicators.calculate_bands(
                self.current_trading_pair, current_close)

            if sma.empty or ema.empty:
                self.logger.warning("Not enough data for signal detection")
                return

            # Check for valid trading signals based on position direction
            is_valid_signal, position_type = self.indicators.is_price_outside_bands(
                self.current_trading_pair, current_close)

            # Check if we can open this position based on current positions
            can_open_position = (
                (position_type == "long" and not self.long_position_open) or
                (position_type == "short" and not self.short_position_open)
            )

            if is_valid_signal and can_open_position:
                signal_description = (
                    "LONG Signal: Price closed above bands"
                    if position_type == "long" else
                    "SHORT Signal: Price closed below bands")

                self.logger.info(
                    f"\nValid Trading Signal Detected:\n"
                    f"========================================\n"
                    f"Trading Pair: {self.current_trading_pair}\n"
                    f"Signal Type: {position_type.upper()}\n"
                    f"Description: {signal_description}\n"
                    f"Entry Price: {current_candle['open']:.6f}\n"
                    f"Current Close: {current_close:.6f}\n"
                    f"Candle Confirmed: {bool(current_candle['confirm'])}\n"
                    f"Current Positions: Long={self.long_position_open}, Short={self.short_position_open}\n"
                    f"========================================")

                # Execute trade only if candle is confirmed
                if bool(current_candle['confirm']):
                    await self.execute_trade(current_candle, position_type)
                else:
                    self.logger.info(
                        "Waiting for candle confirmation before executing trade"
                    )
            elif is_valid_signal and not can_open_position:
                self.logger.info(
                    f"Signal detected but position already exists in {position_type} direction. Skipping trade."
                )

        except Exception as e:
            self.logger.error(f"Error checking signals: {str(e)}")
            self.logger.exception("Full traceback:")

    async def execute_trade(self, candle: pd.Series, position_type: str):
        """Execute trade with position sizing and risk management"""
        try:
            entry_price = float(candle['open'])
            self.logger.info(
                f"\nPreparing Trade Execution:\n"
                f"========================================\n"
                f"Trading Pair: {self.current_trading_pair}\n"
                f"Position Type: {position_type}\n"
                f"Entry Price: ${entry_price:.6f}\n"
                f"Target Margin: ${self.config.POSITION_SIZE:.2f}\n"
                f"Leverage: {self.config.LEVERAGE}x\n"
                f"Current Position State: Long={self.long_position_open}, Short={self.short_position_open}\n"
                f"========================================")

            # Verify we can open this position
            if (position_type == "long" and self.long_position_open) or \
               (position_type == "short" and self.short_position_open):
                self.logger.warning(f"Cannot open {position_type} position - already have one open")
                return

            # Calculate position size based on entry price
            position_info = self.risk_manager.calculate_position_size(entry_price)

            if not position_info:
                self.logger.error("Trade rejected: Cannot calculate valid position size")
                return

            # Calculate stop loss using real market data
            stop_loss = await self.risk_manager.calculate_stop_loss(
                entry_price,
                position_type
            )

            if not stop_loss:
                self.logger.error("Trade rejected: Cannot calculate valid stop loss")
                return

            # Calculate take profit using stop loss level
            take_profit = self.risk_manager.calculate_take_profit(
                entry_price,
                stop_loss,
                position_type
            )

            if not take_profit:
                self.logger.error("Trade rejected: Cannot calculate valid take profit")
                return

            # Get current market price for TPSL validation
            current_price = await self.risk_manager.get_latest_mark_price()
            if not current_price:
                self.logger.error("Trade rejected: Cannot fetch current market price")
                return

            # Validate and adjust TPSL prices if needed
            take_profit, stop_loss = self.risk_manager.validate_tpsl_prices(
                entry_price, take_profit, stop_loss, position_type, current_price
            )

            # Combined market order with TPSL
            entry_order = {
                "instId": self.current_trading_pair,
                "marginMode": position_info["marginMode"],
                "side": "buy" if position_type == "long" else "sell",
                "orderType": "market",
                "size": position_info["size"],
                "lever": position_info["leverage"],
                "tpTriggerPrice": str(take_profit),
                "tpOrderPrice": "-1",  # Market price execution for TP
                "slTriggerPrice": str(stop_loss),
                "slOrderPrice": "-1",  # Market price execution for SL
                "tpTriggerPxType": "mark",  # Use mark price for trigger
                "slTriggerPxType": "mark"   # Use mark price for trigger
            }

            # Log entry order details
            self.logger.info(
                f"\nPlacing Combined Entry Order with TPSL:\n"
                f"========================================\n"
                f"Order Details: {entry_order}\n"
                f"========================================")

            entry_order_response = await self.place_order(entry_order, "/api/v1/trade/order")

            if not entry_order_response or entry_order_response.get('code') != '0':
                self.logger.error(
                    f"\nEntry Order Failed:\n"
                    f"========================================\n"
                    f"Response: {entry_order_response}\n"
                    f"Order Parameters: {entry_order}\n"
                    f"========================================")
                return

            # Verify position was opened successfully
            current_position = await self.risk_manager.get_current_positions()
            await self.handle_position_updates(current_position)

            # Log final trade status
            self.logger.info(
                f"\nTrade Execution Summary:\n"
                f"========================================\n"
                f"Trading Pair: {self.current_trading_pair}\n"
                f"Entry Order: {'Success' if entry_order_response.get('code') == '0' else 'Failed'}\n"
                f"Position Type: {position_type}\n"
                f"Entry Price: ${entry_price:.6f}\n"
                f"Take Profit: ${take_profit:.6f}\n"
                f"Stop Loss: ${stop_loss:.6f}\n"
                f"Position Size: {position_info['size']}\n"
                f"Position State: Long={self.long_position_open}, Short={self.short_position_open}\n"
                f"========================================")

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            self.logger.exception("Full traceback:")

    async def place_order(self, order: Dict, endpoint: str = "/api/v1/trade/order"):
        """Place order via REST API"""
        try:
            url = f"{self.config.REST_URL}{endpoint}"
            timestamp = Utils.get_timestamp()
            nonce = Utils.get_nonce()

            # Generate signature
            signature = Utils.generate_signature(self.config.API_SECRET,
                                                 "POST", endpoint,
                                                 timestamp, nonce, order)

            headers = {
                "ACCESS-KEY": self.config.API_KEY,
                "ACCESS-SIGN": signature,
                "ACCESS-TIMESTAMP": timestamp,
                "ACCESS-NONCE": nonce,
                "ACCESS-PASSPHRASE": self.config.API_PASSPHRASE,
                "Content-Type": "application/json"
            }

            self.logger.debug(f"Sending order request: {order}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order,
                                        headers=headers) as response:
                    data = await response.json()
                    if data['code'] == '0':
                        self.logger.info(f"Order placed successfully: {data}")
                    else:
                        self.logger.error(f"Order placement failed: {data}")
                    return data

        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None

    async def run(self):
        """Run the trading bot"""
        try:
            await self.initialize()
            self.logger.info(
                f"Trading bot started. Watching {self.config.TRADING_PAIR} on {self.config.TIMEFRAME} timeframe"
            )

            if self.websocket_manager is None:
                raise RuntimeError(
                    "WebSocket manager not properly initialized")

            # Start WebSocket handlers to receive real-time market data
            await asyncio.gather(self.websocket_manager.heartbeat(),
                                 self.websocket_manager.message_handler())

        except Exception as e:
            self.logger.error(f"Bot runtime error: {str(e)}")
        finally:
            if self.websocket_manager:
                await self.websocket_manager.close()


# Move main execution block outside the class
if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())