from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class TradeRecord:
    trading_pair: str
    entry_time: datetime
    exit_time: datetime
    position_type: str  # "long" or "short"
    entry_price: float
    exit_price: float
    take_profit: float
    stop_loss: float
    size: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # "take_profit", "stop_loss", or "manual"
    
@dataclass
class DailyTradeTracker:
    date: datetime = field(default_factory=lambda: datetime.now().date())
    trades: List[TradeRecord] = field(default_factory=list)
    
    def add_closed_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        
    def get_trade_summary(self) -> Dict:
        """Generate summary statistics for the day's trading"""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_pnl": 0.0,
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
                "best_pair": "N/A",
                "worst_pair": "N/A"
            }
            
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        # Get best and worst performing pairs
        pair_performance = {}
        for trade in self.trades:
            if trade.trading_pair not in pair_performance:
                pair_performance[trade.trading_pair] = 0
            pair_performance[trade.trading_pair] += trade.pnl
            
        best_pair = max(pair_performance.items(), key=lambda x: x[1])[0] if pair_performance else "N/A"
        worst_pair = min(pair_performance.items(), key=lambda x: x[1])[0] if pair_performance else "N/A"
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(t.pnl for t in self.trades),
            "average_pnl": sum(t.pnl for t in self.trades) / len(self.trades) if self.trades else 0,
            "best_trade_pnl": max(t.pnl for t in self.trades) if self.trades else 0,
            "worst_trade_pnl": min(t.pnl for t in self.trades) if self.trades else 0,
            "best_pair": best_pair,
            "worst_pair": worst_pair
        }
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade records to DataFrame for easier reporting"""
        if not self.trades:
            return pd.DataFrame()
            
        df = pd.DataFrame([vars(trade) for trade in self.trades])
        # Format columns for better readability
        if not df.empty:
            for col in ['entry_price', 'exit_price', 'take_profit', 'stop_loss', 'pnl']:
                if col in df.columns:
                    df[col] = df[col].map('${:,.6f}'.format)
            if 'pnl_percent' in df.columns:
                df['pnl_percent'] = df['pnl_percent'].map('{:+.2f}%'.format)
        return df
