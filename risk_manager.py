from typing import Dict, Optional, Union, List, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import logging
import aiohttp
from utils import Utils  # For API signing

class RiskManager:
    def __init__(self, config):
        """Initialize RiskManager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize with proper type annotations
        self.contract_value: Optional[Decimal] = None
        self.min_size: Optional[Decimal] = None
        self.lot_size: Optional[Decimal] = None
        self.max_leverage: Optional[Decimal] = None
        self.max_market_size: Optional[Decimal] = None
        self.tick_size: Optional[Decimal] = None
        self.margin_tolerance = Decimal('0.05')  # 5% margin tolerance
        #self.candle_data: List[Dict] = []  # Store recent candle data - Removed as no longer needed

    async def initialize(self):
        """Initialize trading parameters from exchange"""
        try:
            # Fetch instrument info from Blofin API
            url = f"{self.config.REST_URL}/api/v1/market/instruments"
            params = {"instId": self.config.TRADING_PAIR}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()

                    if data['code'] == '0' and data['data']:
                        instrument = data['data'][0]

                        # Store ALL relevant trading parameters
                        self.contract_value = Decimal(str(instrument['contractValue']))
                        self.min_size = Decimal(str(instrument['minSize']))
                        self.lot_size = Decimal(str(instrument['lotSize']))
                        self.max_leverage = Decimal(str(instrument['maxLeverage']))
                        self.max_market_size = Decimal(str(instrument['maxMarketSize']))
                        self.tick_size = Decimal(str(instrument['tickSize']))

                        self.logger.info(
                            f"\nTrading Parameters Loaded for {self.config.TRADING_PAIR}:\n"
                            f"========================================\n"
                            f"Contract Value: {float(self.contract_value):.8f}\n"
                            f"Minimum Size: {float(self.min_size):.8f}\n"
                            f"Lot Size: {float(self.lot_size):.8f}\n"
                            f"Tick Size: {float(self.tick_size):.8f}\n"
                            f"Max Leverage: {float(self.max_leverage):.0f}x\n"
                            f"Max Market Size: {float(self.max_market_size):.8f}\n"
                            f"========================================")

                        # Validate leverage configuration
                        if Decimal(str(self.config.LEVERAGE)) > self.max_leverage:
                            raise ValueError(
                                f"Configured leverage ({self.config.LEVERAGE}x) exceeds "
                                f"maximum allowed ({float(self.max_leverage)}x)")

                    else:
                        raise Exception(f"Failed to fetch instrument info: {data.get('msg', 'Unknown error')}")

        except Exception as e:
            self.logger.error(f"Error initializing trading parameters: {str(e)}")
            raise

    async def get_historical_candles(self) -> Optional[List[Dict]]:
        """Fetch last N candles for stop loss calculation using the configured timeframe"""
        try:
            url = f"{self.config.REST_URL}/api/v1/market/candles"
            params = {
                "instId": self.config.TRADING_PAIR,
                "bar": self.config.TIMEFRAME,  # Use configured timeframe instead of hardcoded "5m"
                "limit": str(self.config.CANDLE_LOOKBACK)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    if data['code'] == '0' and data['data']:
                        # Format: [timestamp, open, high, low, close, volume, ...]
                        candles = data['data']
                        formatted_candles = [
                            {
                                'timestamp': float(candle[0]),
                                'high': float(candle[2]),
                                'low': float(candle[3])
                            }
                            for candle in candles
                        ]
                        
                        self.logger.info(
                            f"Fetched {len(formatted_candles)} candles on {self.config.TIMEFRAME} timeframe for SL calculation"
                        )
                        
                        return formatted_candles
                    else:
                        raise Exception(f"Failed to fetch candles: {data.get('msg', 'Unknown error')}")
        except Exception as e:
            self.logger.error(f"Error fetching historical candles: {str(e)}")
            return None

    def round_to_tick(self, price: float, position_type: str = None) -> str:
        """Round price to valid tick size with position-aware rounding"""
        if not self.tick_size:
            raise ValueError("Tick size not initialized")
        
        # Convert price to Decimal for precise calculation
        price_decimal = Decimal(str(price))
        tick_size = self.tick_size
        
        # Get the tick size precision
        tick_precision = abs(tick_size.as_tuple().exponent)
        
        # Use position-aware rounding
        if position_type == "long":
            # For long positions, round UP for take profits to ensure we get at least the desired percentage
            rounding_mode = ROUND_UP
        elif position_type == "short":
            # For short positions, round DOWN for take profits to ensure we get at least the desired percentage
            rounding_mode = ROUND_DOWN
        else:
            # Default to ROUND_DOWN when position type not specified
            rounding_mode = ROUND_DOWN
        
        # Calculate number of ticks
        ticks = (price_decimal / tick_size).quantize(Decimal('1.'), rounding=rounding_mode)
        
        # Calculate final price
        rounded_price = (ticks * tick_size).quantize(
            tick_size,
            rounding=rounding_mode
        )
        
        # Format with exact precision needed
        # Use string formatting to ensure we keep trailing zeros
        return f"{rounded_price:0.{tick_precision}f}"

    async def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """Calculate stop loss based on highest/lowest wicks of recent candles"""
        try:
            # Fetch historical candles if not already available
            candles = await self.get_historical_candles()
            
            if not candles:
                self.logger.error("Failed to fetch historical candle data")
                raise ValueError("Failed to fetch historical candle data for SL calculation")
            
            # Use the configured candle lookback parameter
            recent_candles = candles[-self.config.CANDLE_LOOKBACK:]
            
            # Get extreme prices from recent candles
            highest_wick = max(candle['high'] for candle in recent_candles)
            lowest_wick = min(candle['low'] for candle in recent_candles)
            
            # Convert buffer to decimal
            buffer_percentage = Decimal(str(self.config.STOP_LOSS_BUFFER)) / Decimal('100')
            
            # Set stop loss based on position type and recent wicks with buffer
            if position_type == "long":
                # For long positions, set SL slightly below the lowest wick
                stop_loss = Decimal(str(lowest_wick)) * (Decimal('1') - buffer_percentage)
            else: # short
                # For short positions, set SL slightly above the highest wick
                stop_loss = Decimal(str(highest_wick)) * (Decimal('1') + buffer_percentage)
            
            # Round to valid tick size
            rounded_sl = self.round_to_tick(float(stop_loss))
            
            # Calculate actual risk percentage for logging
            risk_percentage = abs(float(rounded_sl) - float(entry_price)) / float(entry_price) * 100
            
            self.logger.info(
                f"\nStop Loss Calculation:\n"
                f"========================================\n"
                f"Entry Price: ${float(entry_price):.6f}\n"
                f"Position Type: {position_type}\n"
                f"Candles Used: {len(recent_candles)}\n"
                f"Highest Wick: ${highest_wick:.6f}\n"
                f"Lowest Wick: ${lowest_wick:.6f}\n"
                f"Buffer Applied: {float(buffer_percentage) * 100:.2f}%\n"
                f"Target Risk %: 1.00%\n"
                f"Actual Risk %: {risk_percentage:.2f}%\n"
                f"Raw SL Price: ${float(stop_loss):.6f}\n"
                f"Rounded SL Price: ${float(rounded_sl):.6f}\n"
                f"========================================")
            
            return float(rounded_sl)
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            raise

    def calculate_take_profit(self, entry_price: float, stop_loss: float, position_type: str) -> float:
        """Calculate take profit based on configured percentage from entry price"""
        try:
            # Convert inputs to Decimal for precise calculation
            entry_price_decimal = Decimal(str(entry_price))
            tp_percentage = Decimal(str(self.config.TAKE_PROFIT_PERCENT)) / Decimal('100')  # Convert from percentage to decimal

            # Calculate raw take profit price with exact precision
            if position_type == "long":
                tp_price = entry_price_decimal * (Decimal('1.0') + tp_percentage)
            else:  # short
                tp_price = entry_price_decimal * (Decimal('1.0') - tp_percentage)

            # Round to valid tick size using our helper method with position type
            rounded_tp = self.round_to_tick(float(tp_price), position_type)

            # Calculate reward percentage for logging
            reward_percentage = abs(float(rounded_tp) - float(entry_price_decimal)) / float(entry_price_decimal) * 100

            self.logger.info(
                f"\nTake Profit Calculation:\n"
                f"========================================\n"
                f"Entry Price: ${float(entry_price_decimal):.6f}\n"
                f"Position Type: {position_type}\n"
                f"Target TP %: {float(tp_percentage) * 100:.2f}%\n"
                f"Raw TP Price: ${float(tp_price):.6f}\n"
                f"Actual Reward %: {reward_percentage:.2f}%\n"
                f"Rounded TP Price: ${float(rounded_tp)}\n"
                f"========================================")

            return float(rounded_tp)
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            raise

    async def get_latest_mark_price(self, trading_pair: Optional[str] = None) -> Optional[float]:
        """Get the latest mark price for the trading pair"""
        try:
            # Use provided trading pair or default from config
            pair = trading_pair or self.config.TRADING_PAIR

            url = f"{self.config.REST_URL}/api/v1/market/mark-price"
            params = {"instId": pair}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()

                    if data['code'] == '0' and data['data']:
                        return float(data['data'][0]['markPrice'])
                    else:
                        self.logger.error(f"Failed to fetch mark price for {pair}: {data.get('msg', 'Unknown error')}")
                        return None

        except Exception as e:
            self.logger.error(f"Error fetching mark price for {trading_pair}: {str(e)}")
            return None

    def validate_tpsl_prices(self, entry_price: float, take_profit: float, stop_loss: float, position_type: str, current_price: float) -> Tuple[float, float]:
        """Validate SL prices but don't modify the TP (enforce configured percentage)"""
        try:
            # Skip TP validation completely - keep original calculated TP
            
            # Only validate SL
            if position_type == "long":
                # For long positions, SL must be lower than current price
                if stop_loss >= current_price:
                    # Adjust SL to be 0.5% below current price
                    stop_loss = float(self.round_to_tick(current_price * 0.995))
                    self.logger.warning(f"Adjusted SL to {stop_loss} (0.5% below current price)")
            else:
                # For short positions, SL must be higher than current price
                if stop_loss <= current_price:
                    # Adjust SL to be 0.5% above current price
                    stop_loss = float(self.round_to_tick(current_price * 1.005))
                    self.logger.warning(f"Adjusted SL to {stop_loss} (0.5% above current price)")
            
            return take_profit, stop_loss
        except Exception as e:
            self.logger.error(f"Error validating TPSL prices: {str(e)}")
            raise

    def calculate_position_size(self, current_price: float) -> Optional[Dict]:
        """Calculate position size based on USDT margin value and risk parameters"""
        try:
            # Guard clause to ensure required parameters are initialized
            if not all([
                isinstance(self.contract_value, Decimal),
                isinstance(self.min_size, Decimal),
                isinstance(self.lot_size, Decimal),
                isinstance(self.max_leverage, Decimal),
                isinstance(self.max_market_size, Decimal)
            ]):
                self.logger.error("Trading parameters not initialized")
                return None

            # Convert inputs to Decimal for precise calculation
            target_margin = Decimal(str(self.config.POSITION_SIZE))  # This is in USDT
            price = Decimal(str(current_price))
            leverage = Decimal(str(self.config.LEVERAGE))

            # Calculate position value in USDT (margin * leverage)
            position_value_usdt = target_margin * leverage

            # Calculate actual position size in base currency (contracts)
            # position_size = (position_value_usdt / price) / contract_value
            position_size = position_value_usdt / (price * self.contract_value)

            # Round down to lot size precision
            position_size_rounded = (position_size / self.lot_size).quantize(
                Decimal('1'), rounding=ROUND_DOWN) * self.lot_size

            # Calculate actual margin used after rounding
            actual_position_value = position_size_rounded * price * self.contract_value
            actual_margin = actual_position_value / leverage

            # Add this validation after calculating actual_margin
            if actual_margin > (target_margin * (Decimal('1') + self.margin_tolerance)):
                self.logger.warning(
                    f"Calculated margin ${float(actual_margin):.2f} exceeds target margin ${float(target_margin):.2f} by more than {float(self.margin_tolerance*100)}%"
                )
                # Recalculate position size to match target margin
                adjusted_size = (target_margin * leverage) / (price * self.contract_value)
                # Round down to lot size precision
                position_size_rounded = (adjusted_size / self.lot_size).quantize(
                    Decimal('1'), rounding=ROUND_DOWN) * self.lot_size
                # Recalculate actual margin
                actual_position_value = position_size_rounded * price * self.contract_value
                actual_margin = actual_position_value / leverage
                
                self.logger.info(
                    f"Adjusted position size to {float(position_size_rounded):.8f} contracts with margin ${float(actual_margin):.2f}"
                )

            # Extra safety check for maximum position size
            if position_size_rounded > self.max_market_size:
                self.logger.warning(
                    f"Position size {float(position_size_rounded)} exceeds maximum {float(self.max_market_size)}. "
                    f"Adjusting to max size.")
                position_size_rounded = self.max_market_size

            # Verify minimum size
            if position_size_rounded < self.min_size:
                self.logger.error(
                    f"Calculated position size {float(position_size_rounded)} is below minimum {float(self.min_size)}")
                return None

            self.logger.info(
                f"\nPosition Size Calculation Details:\n"
                f"========================================\n"
                f"Input Parameters:\n"
                f"- Target Margin: ${float(target_margin):.2f} USDT\n"
                f"- Leverage: {int(leverage)}x\n"
                f"- Current Price: ${float(price):.6f}\n"
                f"- Contract Value: {float(self.contract_value):.8f}\n"
                f"\nCalculation Results:\n"
                f"- Position Value (USDT): ${float(position_value_usdt):.2f}\n"
                f"- Raw Position Size: {float(position_size):.8f}\n"
                f"- Rounded Position Size: {float(position_size_rounded):.8f} contracts\n"
                f"- Final Position Value: ${float(actual_position_value):.2f}\n"
                f"- Actual Margin Used: ${float(actual_margin):.2f}\n"
                f"========================================")

            return {
                "size": f"{position_size_rounded:f}",
                "leverage": str(int(leverage)),
                "marginMode": self.config.POSITION_TYPE,
                "actualMargin": float(actual_margin)
            }

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return None

    def validate_position_risks(self, position_info: Dict, stop_loss: float,
                                take_profit: float, entry_price: float) -> bool:
        """Validate if position meets risk management criteria"""
        try:
            size = float(position_info["size"])
            actual_margin = position_info["actualMargin"]

            # Calculate potential loss and profit
            price_movement_loss = abs(entry_price - stop_loss)
            loss_percentage = (price_movement_loss / entry_price) * 100
            max_loss = actual_margin * (loss_percentage / 100)

            price_movement_profit = abs(take_profit - entry_price)
            profit_percentage = (price_movement_profit / entry_price) * 100
            max_profit = actual_margin * (profit_percentage / 100)

            # Maximum allowed loss is the configured risk per trade
            max_allowed_loss = self.config.POSITION_SIZE * self.config.RISK_PER_TRADE

            # Calculate risk-reward ratio
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0

            self.logger.info(
                f"\nRisk Validation:\n"
                f"========================================\n"
                f"Position Size: {size} contracts\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Stop Loss: ${stop_loss:.2f} ({loss_percentage:.2f}%)\n"
                f"Take Profit: ${take_profit:.2f} ({profit_percentage:.2f}%)\n"
                f"Risk:Reward = {risk_reward_ratio:.2f}\n"
                f"Max Loss: ${max_loss:.2f}\n"
                f"Max Allowed Loss: ${max_allowed_loss:.2f}\n"
                f"========================================")

            if max_loss > max_allowed_loss:
                self.logger.warning(
                    f"Position rejected: Max loss (${max_loss:.2f}) "
                    f"exceeds allowed (${max_allowed_loss:.2f})")
                return False

            if risk_reward_ratio < 1.5:
                self.logger.warning(
                    f"Position rejected: Poor risk-reward ratio ({risk_reward_ratio:.2f})")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating position risks: {str(e)}")
            raise

    def calculate_dollar_value(self, size: float, price: float) -> float:
        """Calculate position value in USD"""
        return size * price

    async def get_current_positions(self) -> Optional[Dict]:
        """Fetch current positions from Blofin"""
        try:
            endpoint = "/api/v1/account/positions"
            url = f"{self.config.REST_URL}{endpoint}"
            params = {"instId": self.config.TRADING_PAIR}

            # Generate authentication parameters
            timestamp = Utils.get_timestamp()
            nonce = Utils.get_nonce()

            # For GET requests with query params, include them in the endpoint
            query_string = f"?instId={self.config.TRADING_PAIR}"
            signature_endpoint = f"{endpoint}{query_string}"

            # Generate signature
            signature = Utils.generate_signature(
                self.config.API_SECRET,
                "GET",
                signature_endpoint,
                timestamp,
                nonce,
                None  # No body for GET request
            )

            headers = {
                "ACCESS-KEY": self.config.API_KEY,
                "ACCESS-SIGN": signature,
                "ACCESS-TIMESTAMP": timestamp,
                "ACCESS-NONCE": nonce,
                "ACCESS-PASSPHRASE": self.config.API_PASSPHRASE,
                "Content-Type": "application/json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    data = await response.json()

                    if data['code'] == '0':
                        if not data['data']:
                            self.logger.info(f"No open positions found for {self.config.TRADING_PAIR}")
                            return {
                                'instId': self.config.TRADING_PAIR,
                                'positions': '0',
                                'positionSide': 'net'
                            }  # Return empty position data instead of None

                        positions = data['data']
                        for position in positions:
                            # Calculate margin in USD
                            size = float(position['positions'])
                            mark_price = float(position['markPrice'])
                            leverage = float(position['leverage'])

                            margin_usd = (size * mark_price) / leverage
                            position_value_usd = size * mark_price

                            self.logger.info(
                                f"\nCurrent Position Details:\n"
                                f"========================================\n"
                                f"Position ID: {position['positionId']}\n"
                                f"Instrument: {position['instId']}\n"
                                f"Type: {position['marginMode']}\n"
                                f"Side: {position['positionSide']}\n"
                                f"Size: {position['positions']}\n"
                                f"Average Price: ${float(position['averagePrice']):.6f}\n"
                                f"Mark Price: ${mark_price:.6f}\n"
                                f"Position Value: ${position_value_usd:.2f}\n"
                                f"Margin Used: ${margin_usd:.2f}\n"
                                f"Unrealized PnL: ${float(position['unrealizedPnl']):.2f}\n"
                                f"Margin Ratio: {float(position['marginRatio']):.2f}%\n"
                                f"Leverage: {position['leverage']}x\n"
                                f"========================================")

                        return positions[0] if positions else {
                            'instId': self.config.TRADING_PAIR,
                            'positions': '0',
                            'positionSide': 'net'
                        }
                    else:
                        self.logger.error(f"Failed to fetch positions: {data.get('msg', 'Unknown error')}")
                        return {
                            'instId': self.config.TRADING_PAIR,
                            'positions': '0',
                            'positionSide': 'net'
                        }

        except Exception as e:
            self.logger.error(f"Error fetching positions: {str(e)}")
            return {
                'instId': self.config.TRADING_PAIR,
                'positions': '0',
                'positionSide': 'net'
            }
