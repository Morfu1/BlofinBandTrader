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
from market_scanner import MarketScanner
from risk_manager import RiskManager  # Add missing import

class TradingBot:
    def __init__(self):
        """Initialize bot components and configuration"""
        self.config = Config()
        self.indicators = Indicators(sma_period=self.config.SMA_PERIOD,
                                   ema_period=self.config.EMA_PERIOD,
                                   sl_percentage=1.0)
        self.risk_manager = RiskManager(config=self.config)
        self.market_scanner = MarketScanner(config=self.config)
        self.websocket_manager: Optional[WebSocketManager] = None
        self.candles_df = pd.DataFrame()
        self.position_open = False
        self.logger = logging.getLogger(__name__)
        self.historical_data = {}  # Store historical data for each trading pair

        # Track position direction for each trading pair
        self.positions = {}  # Dictionary to store position states for each pair
        self.current_trading_pairs = []  # List of active trading pairs

        self.logger.info(
            f"Initialized with selection mode: {self.config.COIN_SELECTION_MODE}"
        )

    async def handle_position_updates(self, position_data: Optional[Dict]) -> None:
        """Handle position updates and state management"""
        try:
            # If position_data is None, use empty position data
            if position_data is None:
                position_data = {
                    'instId': self.config.TRADING_PAIR,
                    'positions': '0',
                    'positionSide': 'net'
                }

            trading_pair = position_data.get('instId')
            if not trading_pair or trading_pair not in self.current_trading_pairs:
                return

            # Initialize position state for new trading pair
            if trading_pair not in self.positions:
                self.positions[trading_pair] = {
                    'position_open': False,
                    'long_position_open': False,
                    'short_position_open': False
                }

            # Determine position direction from size
            position_size = float(position_data.get('positions', 0))

            # Update position flags based on position size
            if position_size > 0:  # Long position
                self.positions[trading_pair]['long_position_open'] = True
                self.positions[trading_pair]['short_position_open'] = False
            elif position_size < 0:  # Short position
                self.positions[trading_pair]['long_position_open'] = False
                self.positions[trading_pair]['short_position_open'] = True
            else:  # No position
                self.positions[trading_pair]['long_position_open'] = False
                self.positions[trading_pair]['short_position_open'] = False

            # Update overall position state
            self.positions[trading_pair]['position_open'] = (
                self.positions[trading_pair]['long_position_open'] or 
                self.positions[trading_pair]['short_position_open']
            )

            self.logger.info(
                f"\nPosition State Updated for {trading_pair}:\n"
                f"========================================\n"
                f"Position Size: {position_size}\n"
                f"Long Position: {self.positions[trading_pair]['long_position_open']}\n"
                f"Short Position: {self.positions[trading_pair]['short_position_open']}\n"
                f"Any Position Open: {self.positions[trading_pair]['position_open']}\n"
                f"========================================")

        except Exception as e:
            self.logger.error(f"Error handling position update: {str(e)}")
            self.logger.exception("Full traceback:")

    async def initialize(self):
        """Initialize the trading bot with clean state"""
        Utils.setup_logging()
        self.logger.info("Initializing trading bot...")

        # Get trading pairs based on selection mode
        self.current_trading_pairs = await self.market_scanner.get_trading_pairs()

        # Validate trading pairs
        self.current_trading_pairs = await self.market_scanner.validate_trading_pairs(
            self.current_trading_pairs
        )

        self.logger.info(
            f"Initialized with trading pairs: {self.current_trading_pairs}"
        )

        # Reset state for all pairs
        self.positions = {
            pair: {
                'position_open': False,
                'long_position_open': False,
                'short_position_open': False
            }
            for pair in self.current_trading_pairs
        }
        self.candles_df = pd.DataFrame()
        self.historical_data = {pair: pd.DataFrame() for pair in self.current_trading_pairs}

        # Initialize RiskManager with trading parameters
        await self.risk_manager.initialize()

        # Check for existing positions and update state for each pair
        for pair in self.current_trading_pairs:
            self.config.TRADING_PAIR = pair  # Temporarily set pair for position check
            current_position = await self.risk_manager.get_current_positions()
            await self.handle_position_updates(current_position)

        # Reset indicators state
        self.indicators.reset_state()

        # Load fresh historical data for each pair
        for pair in self.current_trading_pairs:
            self.config.TRADING_PAIR = pair  # Set current pair for data loading
            await self.load_historical_data()

        # Setup WebSocket with all trading pairs
        self.websocket_manager = WebSocketManager(self.config)
        await self.websocket_manager.connect()

        # Register candle update callback for each pair
        for pair in self.current_trading_pairs:
            self.websocket_manager.register_callback(
                'candle',
                lambda data, pair=pair: self.handle_candle_update(data, pair)
            )
            self.logger.info(
                f"Registered candle update callback for {pair} on {self.config.TIMEFRAME} timeframe"
            )

        # Initialize indicators with clean data for each pair
        for pair in self.current_trading_pairs:
            if pair in self.historical_data and not self.historical_data[pair].empty:
                self.indicators.initialize_historical_data(
                    pair, self.historical_data[pair]['close'])

        self.logger.info(
            f"\nInitialization Complete:\n"
            f"========================================\n"
            f"Selection Mode: {self.config.COIN_SELECTION_MODE}\n"
            f"Active Trading Pairs: {self.current_trading_pairs}\n"
            f"Timeframe: {self.config.TIMEFRAME}\n"
            f"WebSocket Status: Connected\n"
            f"Callback Registration: Complete\n"
            f"========================================")

    async def handle_candle_update(self, candle_data: List, trading_pair: str):
        """Handle real-time candlestick updates for a specific trading pair"""
        try:
            # Don't modify global config, use trading_pair parameter instead
            if trading_pair not in self.current_trading_pairs:
                return

            # Extract and validate candle data
            try:
                current_candle = {
                    'ts': float(candle_data[0][0]),
                    'open': float(candle_data[0][1]),
                    'high': float(candle_data[0][2]),
                    'low': float(candle_data[0][3]),
                    'close': float(candle_data[0][4]),
                    'vol': float(candle_data[0][5]),
                    'volCurrency': float(candle_data[0][6]),
                    'volCurrencyQuote': float(candle_data[0][7]),
                    'confirm': bool(int(float(candle_data[0][8])))
                    if len(candle_data[0]) > 8 else False,
                }

                # Get previous candle data
                previous_candle = None
                if len(self.historical_data[trading_pair]) > 0:
                    historical_df = self.historical_data[trading_pair]
                    previous_candle = historical_df[historical_df.index < pd.to_datetime(current_candle['ts'], unit='ms')].iloc[-1]

                # Calculate indicators
                sma, ema = self.indicators.calculate_bands(trading_pair, current_candle['close'])

                if not sma.empty and not ema.empty:
                    current_sma = sma.iloc[-1]
                    current_ema = ema.iloc[-1]
                    band_high = max(current_sma, current_ema)
                    band_low = min(current_sma, current_ema)

                    # Position status determination
                    position_status = "BETWEEN BANDS"
                    if current_candle['close'] > band_high:
                        position_status = "ABOVE BANDS"
                    elif current_candle['close'] < band_low:
                        position_status = "BELOW BANDS"

                    # Format timestamp
                    candle_time = datetime.fromtimestamp(current_candle['ts'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

                    # Check if we have an open position
                    has_position = self.positions[trading_pair]['position_open']
                    position_type = (
                        "LONG" if self.positions[trading_pair]['long_position_open']
                        else "SHORT" if self.positions[trading_pair]['short_position_open']
                        else "NONE"
                    )

                    # Calculate price changes
                    current_change = (
                        (current_candle['close'] - current_candle['open']) /
                        current_candle['open'] * 100
                    )

                    # Previous candle stats if available
                    prev_candle_stats = ""
                    if previous_candle is not None:
                        prev_change = (
                            (float(previous_candle['close']) - float(previous_candle['open'])) /
                            float(previous_candle['open']) * 100
                        )
                        # Calculate previous candle's band position
                        prev_sma, prev_ema = self.indicators.calculate_bands(
                            trading_pair, 
                            float(previous_candle['close'])
                        )
                        if not prev_sma.empty and not prev_ema.empty:
                            prev_band_high = max(prev_sma.iloc[-1], prev_ema.iloc[-1])
                            prev_band_low = min(prev_sma.iloc[-1], prev_ema.iloc[-1])
                            prev_position = "BETWEEN BANDS"
                            if float(previous_candle['close']) > prev_band_high:
                                prev_position = "ABOVE BANDS"
                            elif float(previous_candle['close']) < prev_band_low:
                                prev_position = "BELOW BANDS"

                        prev_time = previous_candle.name.strftime('%H:%M:%S')
                        prev_candle_stats = (
                            f"\nPrevious Candle ({prev_time}):\n"
                            f"   Open: ${float(previous_candle['open']):.6f}\n"
                            f"   Close: ${float(previous_candle['close']):.6f}\n"
                            f"   High: ${float(previous_candle['high']):.6f}\n"
                            f"   Low: ${float(previous_candle['low']):.6f}\n"
                            f"   Change: {prev_change:+.2f}%\n"
                            f"   Position vs Bands: {prev_position}"
                        )

                    # Check for signals only if we don't have a position
                    is_valid_signal = False
                    signal_type = ""
                    signal_status = "⚪ NO NEW SIGNALS"

                    if not has_position and current_candle['confirm']:
                        is_valid_signal, signal_type = self.indicators.is_price_outside_bands(
                            trading_pair,
                            current_candle['close'],
                            current_candle['open']
                        )
                        if is_valid_signal:
                            signal_status = f"🔔 {signal_type.upper()}"

                    # Print formatted status update with clear sections
                    self.logger.info(
                        f"\n{'=' * 100}\n"
                        f"🪙 {trading_pair} Status Update - {candle_time}\n"
                        f"{'=' * 100}\n"
                        f"📊 Current Price Action:\n"
                        f"   Close: ${current_candle['close']:.6f} ({current_change:+.2f}%)\n"
                        f"   Open: ${current_candle['open']:.6f}\n"
                        f"   High: ${current_candle['high']:.6f}\n"
                        f"   Low: ${current_candle['low']:.6f}\n"
                        f"   Volume: {current_candle['vol']:.2f}\n"
                        f"   Status: {'✅ Confirmed' if current_candle['confirm'] else '⏳ Pending'}"
                        f"{prev_candle_stats}\n\n"
                        f"📈 Band Analysis:\n"
                        f"   SMA ({self.indicators.sma_period}): ${current_sma:.6f}\n"
                        f"   EMA ({self.indicators.ema_period}): ${current_ema:.6f}\n"
                        f"   Upper Band: ${band_high:.6f}\n"
                        f"   Lower Band: ${band_low:.6f}\n"
                        f"   Current Price vs Bands: {position_status}\n\n"
                        f"💼 Trading Status:\n"
                        f"   Active Position: {position_type}\n"
                        f"   Signal Analysis: {signal_status if not has_position else '🔒 Position Open - Not Scanning'}\n"
                        f"   {'🎯 Valid Entry Point!' if is_valid_signal and not has_position else '📈 Managing Position...' if has_position else '⏳ Waiting for Setup...'}\n"
                        f"{'=' * 100}")

                    # Process signal if valid and candle is confirmed
                    if is_valid_signal and current_candle['confirm'] and not has_position:
                        await self.check_signals(pd.Series(current_candle), trading_pair)

            except (IndexError, ValueError) as e:
                self.logger.error(f"Invalid candle data format for {trading_pair}: {str(e)}")
                return

        except Exception as e:
            self.logger.error(f"Error handling candle update for {trading_pair}: {str(e)}")
            self.logger.exception("Full traceback:")

    async def check_signals(self, current_candle: pd.Series, trading_pair: str) -> None:
        """Check for trading signals"""
        try:
            # Filter candles for current trading pair BEFORE any calculations
            pair_candles = self.historical_data[trading_pair].copy()

            # Only proceed if we have enough data and candle is confirmed
            if len(pair_candles) < 2 or not current_candle['confirm']:
                return

            # Calculate indicators with the current candle's close price
            current_close = float(current_candle['close'])
            current_open = float(current_candle['open'])
            sma, ema = self.indicators.calculate_bands(
                trading_pair, current_close)

            if sma.empty or ema.empty:
                self.logger.warning(f"Not enough data for {trading_pair} signal detection")
                return

            # Get latest position status before making trade decisions
            self.config.TRADING_PAIR = trading_pair  # Set current pair
            current_position = await self.risk_manager.get_current_positions()
            await self.handle_position_updates(current_position)

            # Log current position state before signal check
            self.logger.info(
                f"\nCurrent Position State Before Signal Check for {trading_pair}:\n"
                f"========================================\n"
                f"Long Position Open: {self.positions[trading_pair]['long_position_open']}\n"
                f"Short Position Open: {self.positions[trading_pair]['short_position_open']}\n"
                f"Any Position Open: {self.positions[trading_pair]['position_open']}\n"
                f"========================================")

            # Check for valid trading signals based on position direction
            is_valid_signal, position_type = self.indicators.is_price_outside_bands(
                trading_pair,
                current_close,
                current_open
            )

            # Check if we can open this position based on current positions
            can_open_position = (
                (position_type == "long" and not self.positions[trading_pair]['long_position_open']) or
                (position_type == "short" and not self.positions[trading_pair]['short_position_open'])
            )

            if is_valid_signal and can_open_position:
                signal_description = (
                    "LONG Signal: Price movement from candle open to close indicates long entry"
                    if position_type == "long" else
                    "SHORT Signal: Price movement from candle open to close indicates short entry")

                self.logger.info(
                    f"\nValid Trading Signal Detected for {trading_pair}:\n"
                    f"========================================\n"
                    f"Trading Pair: {trading_pair}\n"
                    f"Signal Type: {position_type.upper()}\n"
                    f"Description: {signal_description}\n"
                    f"Open Price: {current_candle['open']:.6f}\n"
                    f"Close Price: {current_close:.6f}\n"
                    f"Candle Confirmed: {bool(current_candle['confirm'])}\n"
                    f"Current Positions: Long={self.positions[trading_pair]['long_position_open']}, Short={self.positions[trading_pair]['short_position_open']}\n"
                    f"========================================")

                # Execute trade only if candle is confirmed
                if bool(current_candle['confirm']):
                    await self.execute_trade(current_candle, position_type, trading_pair)
                else:
                    self.logger.info(
                        f"Waiting for {trading_pair} candle confirmation before executing trade"
                    )
            elif is_valid_signal and not can_open_position:
                self.logger.info(
                    f"Signal detected for {trading_pair} but position already exists in {position_type} direction. Skipping trade."
                )

        except Exception as e:
            self.logger.error(f"Error checking signals for {trading_pair}: {str(e)}")
            self.logger.exception("Full traceback:")

    async def execute_trade(self, candle: pd.Series, position_type: str, trading_pair: str) -> None:
        """Execute trade with position sizing and risk management"""
        try:
            entry_price = float(candle['open'])
            self.logger.info(
                f"\nPreparing Trade Execution for {trading_pair}:\n"
                f"========================================\n"
                f"Trading Pair: {trading_pair}\n"
                f"Position Type: {position_type}\n"
                f"Entry Price: ${entry_price:.6f}\n"
                f"Target Margin: ${self.config.POSITION_SIZE:.2f}\n"
                f"Leverage: {self.config.LEVERAGE}x\n"
                f"Current Position State: Long={self.positions[trading_pair]['long_position_open']}, Short={self.positions[trading_pair]['short_position_open']}\n"
                f"========================================")

            # Verify we can open this position
            if (position_type == "long" and self.positions[trading_pair]['long_position_open']) or \
               (position_type == "short" and self.positions[trading_pair]['short_position_open']):
                self.logger.warning(f"Cannot open {position_type} position for {trading_pair} - already have one open")
                return

            # Calculate position size based on entry price
            position_info = self.risk_manager.calculate_position_size(entry_price)

            if not position_info:
                self.logger.error(f"Trade for {trading_pair} rejected: Cannot calculate valid position size")
                return

            # Calculate stop loss using real market data
            stop_loss = await self.risk_manager.calculate_stop_loss(
                entry_price,
                position_type
            )

            if not stop_loss:
                self.logger.error(f"Trade for {trading_pair} rejected: Cannot calculate valid stop loss")
                return

            # Calculate take profit using stop loss level
            take_profit = self.risk_manager.calculate_take_profit(
                entry_price,
                stop_loss,
                position_type
            )

            if not take_profit:
                self.logger.error(f"Trade for {trading_pair} rejected: Cannot calculate valid take profit")
                return

            # Get current market price for TPSL validation
            current_price = await self.risk_manager.get_latest_mark_price(trading_pair)
            if not current_price:
                self.logger.error(f"Trade for {trading_pair} rejected: Cannot fetch current market price")
                return

            # Validate and adjust TPSL prices if needed
            take_profit, stop_loss = self.risk_manager.validate_tpsl_prices(
                entry_price, take_profit, stop_loss, position_type, current_price
            )

            # Combined market order with TPSL
            entry_order = {
                "instId": trading_pair,
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
                "slTriggerPxType": "mark"  # Use mark price for trigger
            }

            # Log entry order details
            self.logger.info(
                f"\nPlacing Combined Entry Order with TPSL for {trading_pair}:\n"
                f"========================================\n"
                f"Order Details: {entry_order}\n"
                f"========================================")

            entry_order_response = await self.place_order(entry_order, "/api/v1/trade/order")

            if not entry_order_response or entry_order_response.get('code') != '0':
                self.logger.error(
                    f"\nEntry Order for {trading_pair} Failed:\n"
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
                f"\nTrade Execution Summary for {trading_pair}:\n"
                f"========================================\n"
                f"Trading Pair: {trading_pair}\n"
                f"Entry Order: {'Success' if entry_order_response.get('code') == '0' else 'Failed'}\n"
                f"Position Type: {position_type}\n"
                f"Entry Price: ${entry_price:.6f}\n"
                f"Take Profit: ${take_profit:.6f}\n"
                f"Stop Loss: ${stop_loss:.6f}\n"
                f"Position Size: {position_info['size']}\n"
                f"Position State: Long={self.positions[trading_pair]['long_position_open']}, Short={self.positions[trading_pair]['short_position_open']}\n"
                f"========================================")

        except Exception as e:
            self.logger.error(f"Error executing trade for {trading_pair}: {str(e)}")
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

    async def load_historical_data(self):
        """Load initial historical candlestick data"""
        url = f"{self.config.REST_URL}/api/v1/market/candles"
        params = {
            "instId": self.config.TRADING_PAIR,
            "bar": self.config.TIMEFRAME,  # Use timeframe from config
            "limit": "100"
        }

        self.logger.info(
            f"Loading historical data for {self.config.TRADING_PAIR} on {self.config.TIMEFRAME} timeframe"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if data['code'] == '0':
                    # Clear existing data to prevent mixing with old coin data
                    if self.config.TRADING_PAIR in self.historical_data:
                        self.historical_data[self.config.TRADING_PAIR] = pd.DataFrame()

                    # Convert data to DataFrame
                    candles = data['data']
                    df = pd.DataFrame(candles,
                                      columns=[
                                          'ts', 'open', 'high', 'low', 'close',
                                          'vol', 'volCurrency',
                                          'volCurrencyQuote', 'confirm'
                                      ])

                    # Add trading pair column for data isolation
                    df['trading_pair'] = self.config.TRADING_PAIR

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
                    self.historical_data[self.config.TRADING_PAIR] = df

                    # Initialize historical data for indicators with sorted data
                    self.indicators.initialize_historical_data(
                        self.config.TRADING_PAIR, self.historical_data[self.config.TRADING_PAIR]['close'])

                    # Format dates for logging
                    start_date = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
                    end_date = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')

                    self.logger.info(
                        f"\nHistorical Data Loaded for {self.config.TRADING_PAIR}:\n"
                        f"========================================\n"
                        f"Trading Pair: {self.config.TRADING_PAIR}\n"
                        f"Timeframe: {self.config.TIMEFRAME}\n"
                        f"Number of candles: {len(df)}\n"
                        f"Price range: ${df['close'].min():.6f} - ${df['close'].max():.6f}\n"
                        f"Time range: {start_date} - {end_date}\n"
                        f"DataFrame Memory Usage: {self.historical_data[self.config.TRADING_PAIR].memory_usage().sum() / 1024:.2f} KB\n"
                        f"========================================")
                else:
                    raise Exception(
                        f"Failed to load historical data for {self.config.TRADING_PAIR}: {data['msg']}")

    async def run(self):
        """Run the trading bot"""
        try:
            await self.initialize()
            self.logger.info(
                f"Trading bot started. Watching {self.current_trading_pairs} on {self.config.TIMEFRAME} timeframe"
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