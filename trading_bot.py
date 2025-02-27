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

        # Load fresh historical data for each pair - with retry logic
        for pair in self.current_trading_pairs:
            self.config.TRADING_PAIR = pair  # Set current pair for data loading
            retry_count = 3
            success = False
            
            while retry_count > 0 and not success:
                try:
                    await self.load_historical_data()
                    # Check if we got valid data
                    if (pair in self.historical_data and 
                        not self.historical_data[pair].empty and 
                        len(self.historical_data[pair]) > 10):  # At least 10 candles
                        success = True
                        self.logger.info(f"Successfully loaded historical data for {pair}")
                    else:
                        self.logger.warning(f"Insufficient historical data for {pair}, retrying...")
                        retry_count -= 1
                        await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error loading historical data for {pair}: {str(e)}")
                    retry_count -= 1
                    await asyncio.sleep(1)
            
            if not success:
                self.logger.error(f"Could not load historical data for {pair} after multiple attempts")

        # Setup WebSocket with all trading pairs
        self.websocket_manager = WebSocketManager(self.config)
        await self.websocket_manager.connect()

        # Register candle update callback for each pair
        for pair in self.current_trading_pairs:
            self.websocket_manager.register_callback(
                'candle',
                self.handle_candle_update  # Pass the method reference directly
            )
            self.logger.info(
                f"Registered candle update callback for all pairs on {self.config.TIMEFRAME} timeframe"
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
            # Don't modify global config, use trading_pair parameter directly
            if trading_pair not in self.current_trading_pairs:
                self.logger.warning(f"Received candle for trading pair not in current list: {trading_pair}")
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
                    'confirm': bool(int(float(candle_data[0][8]))) if len(candle_data[0]) > 8 else False,
                }
                
                # Update our internal historical data structure with the new candle
                dt = pd.to_datetime(current_candle['ts'], unit='ms')
                if trading_pair in self.historical_data:
                    if len(self.historical_data[trading_pair]) > 0:
                        last_timestamp = self.historical_data[trading_pair].index[-1]
                        
                        if dt == last_timestamp:
                            # Update existing candle
                            for key, value in current_candle.items():
                                if key != 'ts':  # Skip timestamp as it's the index
                                    if key == 'confirm':
                                        # Convert to boolean
                                        self.historical_data[trading_pair].loc[dt, key] = bool(value)
                                    else:
                                        self.historical_data[trading_pair].loc[dt, key] = value
                        else:
                            # Add new candle
                            new_data = pd.DataFrame({k: [v] for k, v in current_candle.items() if k != 'ts'}, index=[dt])
                            self.historical_data[trading_pair] = pd.concat([self.historical_data[trading_pair], new_data])
                    else:
                        # First data point for this pair
                        new_data = pd.DataFrame({k: [v] for k, v in current_candle.items() if k != 'ts'}, index=[dt])
                        self.historical_data[trading_pair] = new_data
                else:
                    # Initialize historical data for this pair if it doesn't exist
                    self.historical_data[trading_pair] = pd.DataFrame()
                    new_data = pd.DataFrame({k: [v] for k, v in current_candle.items() if k != 'ts'}, index=[dt])
                    self.historical_data[trading_pair] = new_data
                
                # Only log candle updates if explicitly enabled or if it's a confirmed candle
                should_log = (
                    self.config.LOG_CANDLE_UPDATES or 
                    (current_candle['confirm'] and self.config.LOG_CONFIRMED_CANDLES_ONLY)
                )
                
                # Update indicators
                self.indicators.update_real_time_data(
                    trading_pair, 
                    current_candle['ts'], 
                    current_candle['close']
                )
                
                # Calculate indicators
                sma, ema = self.indicators.calculate_bands(trading_pair, current_candle['close'])

                if not sma.empty and not ema.empty:
                    current_sma = sma.iloc[-1]
                    current_ema = ema.iloc[-1]
                    band_high = max(current_sma, current_ema)
                    band_low = min(current_sma, current_ema)

                    # Position status determination
                    has_position = self.positions[trading_pair]['position_open']
                    position_type = "NONE"
                    if self.positions[trading_pair]['long_position_open']:
                        position_type = "LONG"
                    elif self.positions[trading_pair]['short_position_open']:
                        position_type = "SHORT"

                    # Format candle time for display
                    candle_time = datetime.fromtimestamp(current_candle['ts'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

                    # Check for signals only if we don't have a position
                    is_valid_signal = False
                    signal_type = ""

                    if not has_position and current_candle['confirm']:
                        is_valid_signal, signal_type = self.indicators.is_price_outside_bands(
                            trading_pair,
                            current_candle['close'],
                            current_candle['open']
                        )
                    
                    # Only log if it's a signal, confirmed candle, or logging is explicitly enabled
                    if is_valid_signal or should_log:
                        signal_status = f"🔔 {signal_type.upper()}" if is_valid_signal else "⚪ NO NEW SIGNALS"
                        
                        # Determine price position relative to bands
                        price_position = ""
                        if current_candle['close'] > band_high:
                            price_position = "📈 ABOVE BANDS"
                        elif current_candle['close'] < band_low:
                            price_position = "📉 BELOW BANDS"
                        else:
                            price_position = "⚖️ BETWEEN BANDS"
                        
                        # Print formatted status update with clear sections
                        self.logger.info(
                            f"\n{'=' * 50}\n"
                            f"🪙 {trading_pair} - {candle_time} - {'✅ CONFIRMED' if current_candle['confirm'] else '⏳ UPDATING'}\n"
                            f"{'=' * 50}\n"
                            f"💰 Price Action: Open: {current_candle['open']:.6f}, Close: {current_candle['close']:.6f} ({(current_candle['close']-current_candle['open'])/current_candle['open']*100:+.2f}%)\n"
                            f"📊 Bands: High: {band_high:.6f}, Low: {band_low:.6f}\n"
                            f"🔍 Status: {price_position}\n"
                            f"📈 Position: {position_type}\n"
                            f"🔔 Signal: {signal_status}\n"
                            f"{'=' * 50}"
                        )
                        
                        # Only log detailed analysis if in DEBUG mode
                        if self.logger.level <= logging.DEBUG:
                            self.logger.debug(
                                f"\n📈 Detailed Analysis for {trading_pair}:\n"
                                f" Open: {current_candle['open']:.6f}\n"
                                f" Close: {current_candle['close']:.6f}\n"
                                f" Volume: {current_candle['vol']:.2f}\n"
                                f" SMA: {current_sma:.6f}\n"
                                f" EMA: {current_ema:.6f}\n"
                                f" Band High: {band_high:.6f}\n"
                                f" Band Low: {band_low:.6f}"
                            )

                    # Process signal if valid and candle is confirmed
                    if is_valid_signal and current_candle['confirm'] and not has_position:
                        # Temporarily set the trading pair for the execution
                        original_pair = self.config.TRADING_PAIR
                        self.config.TRADING_PAIR = trading_pair
                        await self.check_signals(pd.Series(current_candle), trading_pair)
                        self.config.TRADING_PAIR = original_pair

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

            # Ensure risk manager is initialized with correct trading pair parameters
            self.config.TRADING_PAIR = trading_pair
            await self.risk_manager.initialize()

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
            "bar": self.config.TIMEFRAME,
            "limit": str(self.config.HISTORICAL_CANDLE_LIMIT)  # Use config parameter instead of hardcoded value
        }
        
        self.logger.info(
            f"Loading historical data for {self.config.TRADING_PAIR} on {self.config.TIMEFRAME} timeframe"
        )
        
        try:
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
                            'confirm': bool,
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
                        self.logger.error(f"Failed to load historical data: {data.get('msg', 'Unknown error')}")
                        # Initialize with empty DataFrame to prevent errors
                        self.historical_data[self.config.TRADING_PAIR] = pd.DataFrame(columns=['close'])
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            # Initialize with empty DataFrame to prevent errors
            self.historical_data[self.config.TRADING_PAIR] = pd.DataFrame(columns=['close'])

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
            await asyncio.gather(
                self.websocket_manager.heartbeat(),
                self.websocket_manager.message_handler(),
                self.websocket_manager.connection_health_monitor()  # Add this new task
            )

        except Exception as e:
            self.logger.error(f"Bot runtime error: {str(e)}")
        finally:
            if self.websocket_manager:
                await self.websocket_manager.close()


# Move main execution block outside the class
if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())
