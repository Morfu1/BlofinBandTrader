import logging
import time
from decimal import Decimal
from typing import Optional, Dict

class BandBasedStopLossManager:
    def __init__(self, config, risk_manager, indicators):
        """Initialize BandBasedStopLossManager with configuration"""
        self.config = config
        self.risk_manager = risk_manager
        self.indicators = indicators
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Use dollar value instead of percentage
        self.activation_profit_usd = Decimal(str(getattr(config, 'BAND_SL_ACTIVATION_PROFIT', 2.0)))
        self.band_buffer_pct = Decimal(str(getattr(config, 'BAND_SL_BUFFER', 0.2)))
        self.enabled = getattr(config, 'BAND_SL_ENABLED', True)
        
        # Tracking state for each position
        self.position_states = {}  # Dict to track state of each position

    async def initialize_position_tracking(self, trading_pair: str, position_type: str,
                               entry_price: float, initial_sl: float) -> None:
        """Initialize tracking for a new position"""
        self.position_states[trading_pair] = {
            'entry_price': Decimal(str(entry_price)),
            'position_type': position_type,  # Changed from position_side to position_type
            'initial_sl': Decimal(str(initial_sl)),  # Initial stop loss
            'current_sl': Decimal(str(initial_sl)),  # Current stop loss (will be updated)
            'band_sl_activated': False,  # Whether band-based SL has been activated
            'last_sl_update_time': time.time(),  # Timestamp of last SL update
            'last_candle_ts': 0,  # Track last processed candle timestamp
            'highest_unrealized_pnl': Decimal('0'),  # Track highest dollar PnL reached
            'break_even_set': False  # Track if we've moved to break-even
        }
        
        self.logger.info(
            f"\nBand-Based Stop Loss Tracking Initialized for {trading_pair}:\n"
            f"========================================\n"
            f"Position Type: {position_type}\n"
            f"Entry Price: ${float(entry_price):.8f}\n"
            f"Initial Stop Loss: ${float(initial_sl):.8f}\n"
            f"Activation Profit Threshold: ${float(self.activation_profit_usd):.2f}\n"
            f"Band Buffer: {float(self.band_buffer_pct)}%\n"
            f"========================================")

    async def process_candle_update(self, trading_pair, candle_data, bot):
        """Process candle updates to update stop loss if needed"""
        # Skip if no position for this trading pair
        if trading_pair not in self.position_states:
            return
            
        state = self.position_states[trading_pair]
        position_type = state['position_type']
        
        # Get current market price - handle both array and dictionary formats
        if isinstance(candle_data, dict):
            current_price = float(candle_data['close'])
        else:
            current_price = float(candle_data[4])  # Close price
        
        # Get current position details
        position = await bot.risk_manager.get_current_positions()
        if not position or float(position.get('positions', 0)) == 0:
            self.reset_position_state(trading_pair)
            return
            
        # Calculate profit
        entry_price = state['entry_price']
        size = abs(float(position.get('positions', 0)))
        
        # Calculate unrealized P&L in dollars
        if position_type == 'long':
            dollar_pnl = (current_price - entry_price) * size
        else:
            dollar_pnl = (entry_price - current_price) * size
        
        # Update state with profit information
        self.position_states[trading_pair]['dollar_pnl'] = dollar_pnl
        
        # Log profit status for debugging
        self.logger.info(
            f"\nPosition P&L Status:\n"
            f"Trading Pair: {trading_pair}\n"
            f"Position Type: {position_type}\n"
            f"Entry Price: ${entry_price:.6f}\n"
            f"Current Price: ${current_price:.6f}\n"
            f"Position Size: {size}\n"
            f"Current P&L: ${dollar_pnl:.2f}\n"
            f"Activation Threshold: ${bot.config.BAND_SL_ACTIVATION_PROFIT:.2f}\n"
            f"Break-Even Set: {state['break_even_set']}"
        )
        
        # First check: Move to break-even when profit reaches activation threshold
        activation_profit_usd = bot.config.BAND_SL_ACTIVATION_PROFIT
        if dollar_pnl >= float(activation_profit_usd) and not state['break_even_set']:
            # Move stop loss to break-even (entry price with small buffer)
            if position_type == 'long':
                # For long positions, add a small buffer below entry
                buffer_amount = entry_price * (bot.config.STOP_LOSS_BUFFER / 100)
                new_sl = entry_price - buffer_amount
            else:
                # For short positions, add a small buffer above entry
                buffer_amount = entry_price * (bot.config.STOP_LOSS_BUFFER / 100)
                new_sl = entry_price + buffer_amount
                
            # Update stop loss
            self.logger.info(f"Attempting to move SL to break-even at ${new_sl:.6f}")
            success = await bot.update_stop_loss(trading_pair, position_type, new_sl)
            
            if success:
                self.position_states[trading_pair]['break_even_set'] = True
                self.position_states[trading_pair]['current_stop_loss'] = new_sl
                self.position_states[trading_pair]['band_tracking_active'] = True
                self.logger.info(f"Successfully moved SL to break-even at ${new_sl:.6f}")
            else:
                self.logger.error(f"Failed to move SL to break-even")
                    
        # Second check: If already at break-even, track bands
        elif state.get('band_tracking_active', False) and state.get('break_even_set', False):
            # Calculate bands
            sma, ema = bot.indicators.calculate_bands(trading_pair)
            
            if len(sma) > 0 and len(ema) > 0:
                upper_band = max(sma.iloc[-1], ema.iloc[-1])
                lower_band = min(sma.iloc[-1], ema.iloc[-1])
                
                # Add buffer to bands
                buffer_pct = bot.config.BAND_SL_BUFFER / 100
                upper_band_buffer = upper_band * (1 + buffer_pct)
                lower_band_buffer = lower_band * (1 - buffer_pct)
                
                # Get current SL
                current_sl = state.get('current_stop_loss', 0)
                
                # Determine new stop loss based on position type
                if position_type == 'long':
                    # For long positions, use lower band as stop loss
                    if lower_band_buffer > current_sl:
                        new_sl = lower_band_buffer
                        
                        # Update stop loss
                        self.logger.info(f"Attempting to update band-based SL to ${new_sl:.6f}")
                        success = await bot.update_stop_loss(trading_pair, position_type, new_sl)
                        
                        if success:
                            self.position_states[trading_pair]['current_stop_loss'] = new_sl
                            self.logger.info(f"Successfully updated band-based SL to ${new_sl:.6f}")
                else:  # short position
                    # For short positions, use upper band as stop loss
                    if upper_band_buffer < current_sl or current_sl == 0:
                        new_sl = upper_band_buffer
                        
                        # Update stop loss
                        self.logger.info(f"Attempting to update band-based SL to ${new_sl:.6f}")
                        success = await bot.update_stop_loss(trading_pair, position_type, new_sl)
                        
                        if success:
                            self.position_states[trading_pair]['current_stop_loss'] = new_sl
                            self.logger.info(f"Successfully updated band-based SL to ${new_sl:.6f}")


    def remove_position_tracking(self, trading_pair: str) -> None:
        """Remove tracking when position is closed"""
        if trading_pair in self.position_states:
            self.logger.info(f"Removing band-based stop tracking for closed position: {trading_pair}")
            del self.position_states[trading_pair]
