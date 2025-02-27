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
        
        # Use configuration parameters if available
        self.activation_profit_pct = Decimal(str(getattr(config, 'BAND_SL_ACTIVATION_PROFIT', 2.0)))
        self.band_buffer_pct = Decimal(str(getattr(config, 'BAND_SL_BUFFER', 0.2)))
        self.enabled = getattr(config, 'BAND_SL_ENABLED', True)
        
        # Tracking state for each position
        self.position_states = {}  # Dict to track state of each position
        
    async def initialize_position_tracking(self, trading_pair: str, position_side: str, 
                                         entry_price: float, initial_sl: float) -> None:
        """Initialize tracking for a new position"""
        self.position_states[trading_pair] = {
            'entry_price': Decimal(str(entry_price)),
            'position_side': position_side,            # 'long' or 'short'
            'initial_sl': Decimal(str(initial_sl)),    # Initial stop loss
            'current_sl': Decimal(str(initial_sl)),    # Current stop loss (will be updated)
            'band_sl_activated': False,                # Whether band-based SL has been activated
            'last_sl_update_time': time.time(),        # Timestamp of last SL update
            'last_candle_ts': 0,                       # Track last processed candle timestamp
            'highest_pnl_pct': Decimal('0')            # Track highest PnL percentage reached
        }
        
        self.logger.info(
            f"\nBand-Based Stop Loss Tracking Initialized for {trading_pair}:\n"
            f"========================================\n"
            f"Position Side: {position_side}\n"
            f"Entry Price: ${float(entry_price):.8f}\n"
            f"Initial Stop Loss: ${float(initial_sl):.8f}\n"
            f"Activation Profit Threshold: {float(self.activation_profit_pct)}%\n"
            f"Band Buffer: {float(self.band_buffer_pct)}%\n"
            f"========================================")
    
    async def process_candle_update(self, trading_pair: str, 
                                  candle_data: Dict, 
                                  current_pnl_pct: float) -> Optional[float]:
        """Process a completed candle and adjust stop loss if needed"""
        if trading_pair not in self.position_states:
            return None  # No position being tracked for this pair
        
        state = self.position_states[trading_pair]
        
        # Skip duplicate candle processing
        candle_timestamp = candle_data['ts']
        if candle_timestamp <= state['last_candle_ts']:
            return None
        
        # Update last processed candle timestamp
        state['last_candle_ts'] = candle_timestamp
        
        # Only process confirmed candles
        if not candle_data.get('confirm', False):
            return None
            
        # Convert current PnL percentage to Decimal
        current_pnl_pct = Decimal(str(current_pnl_pct))
        
        # Update highest PnL reached
        if current_pnl_pct > state['highest_pnl_pct']:
            state['highest_pnl_pct'] = current_pnl_pct
            
        # Check if we should activate band-based SL
        if not state['band_sl_activated'] and current_pnl_pct >= self.activation_profit_pct:
            state['band_sl_activated'] = True
            self.logger.info(
                f"\nBand-Based Stop Loss Activated for {trading_pair}:\n"
                f"========================================\n"
                f"Current PnL: {float(current_pnl_pct):.2f}%\n"
                f"Activation Threshold: {float(self.activation_profit_pct)}%\n"
                f"========================================")
        
        # If band-based SL is activated, calculate new SL based on bands
        if state['band_sl_activated']:
            # Calculate indicators to get current bands
            sma, ema = self.indicators.calculate_bands(trading_pair, float(candle_data['close']))
            
            if sma.empty or ema.empty:
                return None  # Not enough data for calculation
                
            # Get current band values
            current_sma = Decimal(str(sma.iloc[-1]))
            current_ema = Decimal(str(ema.iloc[-1]))
            
            # Determine upper and lower bands
            upper_band = max(current_sma, current_ema)
            lower_band = min(current_sma, current_ema)
            
            # Calculate buffer values
            buffer_amount_upper = upper_band * self.band_buffer_pct / Decimal('100')
            buffer_amount_lower = lower_band * self.band_buffer_pct / Decimal('100')
            
            # Calculate new SL based on position type and bands
            if state['position_side'] == 'long':
                # For long positions, SL should be just below the lower band
                new_sl_raw = lower_band - buffer_amount_lower
            else:  # short position
                # For short positions, SL should be just above the upper band
                new_sl_raw = upper_band + buffer_amount_upper
                
            # Only update if the new SL is better than current SL
            # For longs, "better" means higher. For shorts, "better" means lower.
            sl_improvement = False
            if state['position_side'] == 'long' and new_sl_raw > state['current_sl']:
                sl_improvement = True
            elif state['position_side'] == 'short' and new_sl_raw < state['current_sl']:
                sl_improvement = True
                
            # Update SL if there's improvement
            if sl_improvement:
                # Round to valid tick size using risk manager
                new_sl = float(self.risk_manager.round_to_tick(float(new_sl_raw), state['position_side']))
                
                # Update the stop loss
                state['current_sl'] = Decimal(str(new_sl))
                state['last_sl_update_time'] = time.time()
                
                self.logger.info(
                    f"\nBand-Based Stop Loss Updated for {trading_pair}:\n"
                    f"========================================\n"
                    f"Position Side: {state['position_side']}\n"
                    f"Current Close Price: ${float(candle_data['close']):.8f}\n"
                    f"Upper Band: ${float(upper_band):.8f}\n"
                    f"Lower Band: ${float(lower_band):.8f}\n"
                    f"New Stop Loss: ${float(new_sl):.8f}\n"
                    f"Current PnL: {float(current_pnl_pct):.2f}%\n"
                    f"Highest PnL Reached: {float(state['highest_pnl_pct'])}%\n"
                    f"========================================")
                
                # Return the new stop loss so it can be applied
                return new_sl
        
        # Return None if no update is needed
        return None
    
    def remove_position_tracking(self, trading_pair: str) -> None:
        """Remove tracking when position is closed"""
        if trading_pair in self.position_states:
            self.logger.info(f"Removing band-based stop tracking for closed position: {trading_pair}")
            del self.position_states[trading_pair]
