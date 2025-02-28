import json
import sys
import os
import asyncio
import logging
import time
import aiohttp
from decimal import Decimal
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from config import Config
from risk_manager import RiskManager
from indicators import Indicators
from band_based_sl import BandBasedStopLossManager
from utils import Utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_sl_update")

class TestSLUpdate:
    def __init__(self):
        """Initialize test components"""
        self.config = Config()
        self.risk_manager = RiskManager(config=self.config)
        self.indicators = Indicators(
            sma_period=self.config.SMA_PERIOD,
            ema_period=self.config.EMA_PERIOD,
            sl_percentage=self.config.SL_PERCENTAGE
        )
        
        # Initialize band-based stop loss manager
        self.band_sl_manager = BandBasedStopLossManager(
            self.config,
            self.risk_manager,
            self.indicators
        )
        
        self.logger = logger

    async def initialize(self):
        """Initialize components and load trading parameters"""
        self.logger.info("Initializing test environment...")
        await self.risk_manager.initialize()
        
        # Load historical data for indicators
        for pair in self.config.MULTIPLE_COINS:
            self.config.TRADING_PAIR = pair
            try:
                await self.load_historical_data(pair)
            except Exception as e:
                self.logger.error(f"Error loading data for {pair}: {str(e)}")
    
    async def load_historical_data(self, trading_pair):
        """Load historical candlestick data for a trading pair"""
        self.logger.info(f"Loading historical data for {trading_pair}...")
        
        url = f"{self.config.REST_URL}/api/v1/market/candles"
        params = {
            "instId": trading_pair,
            "bar": self.config.TIMEFRAME,
            "limit": str(self.config.HISTORICAL_CANDLE_LIMIT)
        }
        
        import aiohttp
        import pandas as pd
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data['code'] == '0' and data['data']:
                    # Convert data to DataFrame
                    candles = data['data']
                    df = pd.DataFrame(candles, 
                                     columns=['ts', 'open', 'high', 'low', 'close', 
                                              'vol', 'volCurrency', 'volCurrencyQuote', 'confirm'])
                    
                    # Convert types
                    df = df.astype({
                        'ts': float,
                        'open': float,
                        'high': float,
                        'low': float,
                        'close': float,
                        'vol': float,
                        'volCurrency': float,
                        'volCurrencyQuote': float,
                        'confirm': bool
                    })
                    
                    # Set timestamp as index
                    df.index = pd.to_datetime(df['ts'].astype(int), unit='ms')
                    df = df.sort_index()
                    
                    # Initialize indicators with this data
                    self.indicators.initialize_historical_data(trading_pair, df['close'])
                    self.logger.info(f"Loaded {len(df)} candles for {trading_pair}")
                else:
                    self.logger.error(f"Failed to load historical data: {data.get('msg', 'Unknown error')}")
    
    async def get_open_positions(self):
        """Get all open positions from the exchange"""
        self.logger.info("Fetching open positions...")
        positions = []
        
        # Save original trading pair
        original_pair = self.config.TRADING_PAIR
        
        for pair in self.config.MULTIPLE_COINS:
            # Set current pair
            self.config.TRADING_PAIR = pair
            
            position = await self.risk_manager.get_current_positions(log_details=True)
            
            # Check if position exists and has non-zero size
            if position and float(position.get('positions', 0)) != 0:
                positions.append(position)
                self.logger.info(f"Found open position for {pair}: {position['positions']} contracts")
        
        # Restore original trading pair
        self.config.TRADING_PAIR = original_pair
        
        return positions

    
    async def get_algo_orders(self, trading_pair):
        """Get TP/SL orders for a position"""
        try:
            # Save original trading pair
            original_pair = self.config.TRADING_PAIR
            
            # Set current trading pair
            self.config.TRADING_PAIR = trading_pair
            
            # Get current position
            position = await self.risk_manager.get_current_positions(log_details=True)
            
            # Check if position exists and has non-zero size
            if position and float(position.get('positions', 0)) != 0:
                # Get stop loss and take profit from position data
                sl_price = position.get('slTriggerPrice')
                tp_price = position.get('tpTriggerPrice')
                
                if sl_price:
                    return [{
                        'orderId': position.get('positionId'),
                        'sl': sl_price,
                        'tp': tp_price,
                        'state': 'live'
                    }]
            
            # If no position or SL not found, try getting from TPSL orders
            url = f"{self.config.REST_URL}/api/v1/trade/orders-tpsl-pending"
            params = {"instId": trading_pair}
            
            # Generate authentication parameters
            timestamp = Utils.get_timestamp()
            nonce = Utils.get_nonce()
            
            # For GET requests with query params
            query_string = f"?instId={trading_pair}"
            signature_endpoint = f"/api/v1/trade/orders-tpsl-pending{query_string}"
            
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
                    if data['code'] == '0' and data['data']:
                        tpsl_orders = data['data']
                        result = []
                        for order in tpsl_orders:
                            if order['state'] == 'live' and order['instId'] == trading_pair:
                                result.append({
                                    'orderId': order['tpslId'],
                                    'sl': order.get('slTriggerPrice'),
                                    'tp': order.get('tpTriggerPrice'),
                                    'state': order['state']
                                })
                        return result
            
            # Restore original trading pair
            self.config.TRADING_PAIR = original_pair
            
            return []
        except Exception as e:
            self.logger.error(f"Error getting algo orders: {str(e)}")
            
            # Restore original trading pair
            self.config.TRADING_PAIR = original_pair
            
            return []

    
    async def update_stop_loss(self, trading_pair, position_type, new_stop_loss):
        """Update the stop loss for an existing position"""
        self.logger.info(f"Updating stop loss for {trading_pair} to {new_stop_loss}...")
        try:
            # Save original trading pair
            original_pair = self.config.TRADING_PAIR
            # Set current pair
            self.config.TRADING_PAIR = trading_pair
            
            # Get current position
            position = await self.risk_manager.get_current_positions(log_details=True)
            if not position or float(position.get('positions', 0)) == 0:
                self.logger.error(f"No active position found for {trading_pair}")
                self.config.TRADING_PAIR = original_pair
                return False
            
            # Get existing TPSL orders
            algo_orders = await self.get_algo_orders(trading_pair)
            
            # Cancel existing TPSL orders first
            tpsl_ids = []
            for order in algo_orders:
                if order.get('sl') or order.get('tp'):  # If it's a SL or TP order
                    tpsl_ids.append({
                        "instId": trading_pair,
                        "tpslId": order.get('orderId'),  # Make sure this matches your actual order ID field
                        "clientOrderId": ""
                    })
            
            # If we found TPSL orders to cancel
            if tpsl_ids:
                # Cancel existing TPSL orders
                cancel_response = await self.cancel_tpsl_orders(tpsl_ids)
                
                if not cancel_response or cancel_response.get('code') != '0':
                    self.logger.error(f"Failed to cancel existing TPSL orders: {cancel_response}")
                    self.config.TRADING_PAIR = original_pair
                    return False
                
                self.logger.info(f"Successfully cancelled existing TPSL orders")
            
            # Format stop loss price correctly
            new_stop_loss = self.risk_manager.round_to_tick(float(new_stop_loss))
            
            # Create new TPSL order
            position_size = position.get('positions')
            
            # Get existing TP values to preserve them
            current_tp = None
            if position.get('tpTriggerPrice'):
                current_tp = position.get('tpTriggerPrice')
                
            # Create the new TPSL order
            tpsl_order = {
                "instId": trading_pair,
                "marginMode": position.get('marginMode', 'isolated'),
                "positionSide": position.get('positionSide', 'net'),
                "side": "buy" if position_type == "short" else "sell",
                "size": position_size,
                "reduceOnly": "true"  # Important: TP/SL orders should be reduce-only
            }
            
            # Add SL parameters
            tpsl_order["slTriggerPrice"] = str(new_stop_loss)
            tpsl_order["slOrderPrice"] = "-1"  # Market order
            
            # Add TP if it exists
            if current_tp:
                tpsl_order["tpTriggerPrice"] = str(current_tp)
                tpsl_order["tpOrderPrice"] = "-1"  # Market order
                
            # Place the new TPSL order
            url = f"{self.config.REST_URL}/api/v1/trade/order-tpsl"
            timestamp = Utils.get_timestamp()
            nonce = Utils.get_nonce()
            
            # Generate signature
            signature = Utils.generate_signature(
                self.config.API_SECRET,
                "POST",
                "/api/v1/trade/order-tpsl",
                timestamp,
                nonce,
                tpsl_order
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
                async with session.post(url, json=tpsl_order, headers=headers) as response:
                    data = await response.json()
                    
                    # Restore original trading pair
                    self.config.TRADING_PAIR = original_pair
                    
                    if data['code'] == '0':
                        self.logger.info(f"Successfully updated stop loss to ${float(new_stop_loss)}")
                        return True
                    else:
                        self.logger.error(f"Failed to update stop loss: {data}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error updating stop loss: {str(e)}")
            # Restore original trading pair
            self.config.TRADING_PAIR = original_pair
            return False

    async def cancel_tpsl_orders(self, tpsl_ids):
        """Cancel existing TP/SL orders"""
        try:
            url = f"{self.config.REST_URL}/api/v1/trade/cancel-tpsl"
            timestamp = Utils.get_timestamp()
            nonce = Utils.get_nonce()
            
            # Generate signature
            signature = Utils.generate_signature(
                self.config.API_SECRET,
                "POST",
                "/api/v1/trade/cancel-tpsl",
                timestamp,
                nonce,
                tpsl_ids  # This is the body of the request - an array of orders to cancel
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
                async with session.post(url, json=tpsl_ids, headers=headers) as response:
                    data = await response.json()
                    
                    if data['code'] == '0':
                        # This means the API request was successful, but individual orders might still fail
                        # Check individual order results
                        failed_orders = [order for order in data.get('data', []) if order.get('code') != '0']
                        
                        if failed_orders:
                            self.logger.warning(f"Some TPSL cancel orders failed: {failed_orders}")
                        
                        return data
                    else:
                        self.logger.error(f"Failed to send cancel TPSL request: {data}")
                        return None
            
        except Exception as e:
            self.logger.error(f"Error cancelling TPSL orders: {str(e)}")
            return None




  
    async def place_order(self, order, endpoint):
        """Place order via REST API"""
        try:
            url = f"{self.config.REST_URL}{endpoint}"
            timestamp = Utils.get_timestamp()
            nonce = Utils.get_nonce()
            
            # Generate signature
            signature = Utils.generate_signature(
                self.config.API_SECRET,
                "POST", 
                endpoint,
                timestamp, 
                nonce, 
                order
            )
            
            headers = {
                "ACCESS-KEY": self.config.API_KEY,
                "ACCESS-SIGN": signature,
                "ACCESS-TIMESTAMP": timestamp,
                "ACCESS-NONCE": nonce,
                "ACCESS-PASSPHRASE": self.config.API_PASSPHRASE,
                "Content-Type": "application/json"
            }
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order, headers=headers) as response:
                    data = await response.json()
                    
                    if data['code'] == '0':
                        self.logger.info(f"Order placed successfully: {data}")
                    else:
                        self.logger.error(f"Order placement failed: {data}")
                    
                    return data
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    async def run_test(self):
        """Main test function to update SL for profitable positions"""
        try:
            await self.initialize()
            
            # Get open positions
            positions = await self.get_open_positions()
            if not positions:
                self.logger.info("No open positions found")
                return

            self.logger.info(f"Found {len(positions)} open positions")

            # Process each position
            for position in positions:
                trading_pair = position.get('instId')
                position_size = float(position.get('positions', 0))
                position_type = "long" if position_size > 0 else "short"
                entry_price = float(position.get('averagePrice', 0))
                mark_price = float(position.get('markPrice', 0))

                self.logger.info(f"Processing {position_type} position for {trading_pair}")

                # Get unrealized PnL directly from position data
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                is_profitable = unrealized_pnl > 0

                self.logger.info(f"Position {trading_pair} {'is' if is_profitable else 'is not'} in profit: ${unrealized_pnl:.2f}")

                # Check if position meets profit threshold for SL adjustment
                if is_profitable and unrealized_pnl >= self.config.BAND_SL_ACTIVATION_PROFIT:
                    # Save original trading pair
                    original_pair = self.config.TRADING_PAIR
                    
                    # Set current pair
                    self.config.TRADING_PAIR = trading_pair
                    
                    # Get all algo orders for this trading pair
                    algo_orders = await self.get_algo_orders(trading_pair)
                    
                    # Find the stop loss for this position
                    sl_order = None
                    for order in algo_orders:
                        if order.get('sl'):
                            sl_order = order
                            break
                    
                    if sl_order:
                        current_sl = float(sl_order.get('sl', 0))
                        self.logger.info(f"Current stop loss for {trading_pair}: ${current_sl:.6f}")
                        
                        # Initialize position tracking if not already done
                        if trading_pair not in self.band_sl_manager.position_states:
                            await self.band_sl_manager.initialize_position_tracking(
                                trading_pair=trading_pair,
                                position_side=position_type,
                                entry_price=entry_price,
                                initial_sl=current_sl
                            )
                        
                        # Check if we should move to break-even
                        state = self.band_sl_manager.position_states.get(trading_pair, {})
                        if not state.get('break_even_set', False):
                            # Move to break-even
                            if position_type == "long":
                                buffer_amount = entry_price * (self.config.STOP_LOSS_BUFFER / 100)
                                new_sl = entry_price - buffer_amount
                            else:
                                buffer_amount = entry_price * (self.config.STOP_LOSS_BUFFER / 100)
                                new_sl = entry_price + buffer_amount
                            
                            self.logger.info(f"Moving SL to break-even at ${new_sl:.6f}")
                            
                            # Update the stop loss
                            success = await self.update_stop_loss(trading_pair, position_type, new_sl)
                            if success:
                                self.logger.info(f"Successfully updated SL for {trading_pair} to ${new_sl:.6f}")
                                
                                # Update position state
                                if trading_pair in self.band_sl_manager.position_states:
                                    state = self.band_sl_manager.position_states[trading_pair]
                                    state['current_sl'] = Decimal(str(new_sl))
                                    state['break_even_set'] = True
                                    state['last_sl_update_time'] = datetime.now().timestamp()
                            else:
                                self.logger.error(f"Failed to update SL for {trading_pair}")
                        else:
                            # If already at break-even, calculate band-based SL
                            sma, ema = self.indicators.calculate_bands(trading_pair)
                            if len(sma) > 0 and len(ema) > 0:
                                upper_band = max(sma.iloc[-1], ema.iloc[-1])
                                lower_band = min(sma.iloc[-1], ema.iloc[-1])
                                
                                # Add buffer to bands
                                buffer_pct = self.config.BAND_SL_BUFFER / 100
                                upper_band_buffer = upper_band * (1 + buffer_pct)
                                lower_band_buffer = lower_band * (1 - buffer_pct)
                                
                                # Determine new stop loss based on position type
                                new_sl = None
                                if position_type == "long":
                                    # For long positions, use lower band as stop loss
                                    if lower_band_buffer > current_sl:
                                        new_sl = float(lower_band_buffer)
                                else: # short position
                                    # For short positions, use upper band as stop loss
                                    if upper_band_buffer < current_sl or current_sl == 0:
                                        new_sl = float(upper_band_buffer)
                                
                                if new_sl:
                                    # Update the stop loss
                                    success = await self.update_stop_loss(trading_pair, position_type, new_sl)
                                    if success:
                                        self.logger.info(f"Successfully updated band-based SL for {trading_pair} to ${new_sl:.6f}")
                                        
                                        # Update position state
                                        if trading_pair in self.band_sl_manager.position_states:
                                            state = self.band_sl_manager.position_states[trading_pair]
                                            state['current_sl'] = Decimal(str(new_sl))
                                            state['last_sl_update_time'] = datetime.now().timestamp()
                                    else:
                                        self.logger.error(f"Failed to update band-based SL for {trading_pair}")
                    else:
                        self.logger.error(f"No stop loss order found for {trading_pair}")
                    
                    # Restore original trading pair
                    self.config.TRADING_PAIR = original_pair
                else:
                    self.logger.info(f"Position {trading_pair} not in profit or below threshold. Current profit: ${unrealized_pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Error in test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())


# Run the test
if __name__ == "__main__":
    test = TestSLUpdate()
    asyncio.run(test.run_test())
