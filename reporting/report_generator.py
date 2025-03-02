import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional
from .trade_tracker import DailyTradeTracker

class TradingReportGenerator:
    def __init__(self, trade_tracker: DailyTradeTracker):
        self.trade_tracker = trade_tracker
        
    def generate_html_report(self) -> str:
        """Generate a professional HTML report of the day's trading activity"""
        # Get summary statistics
        summary = self.trade_tracker.get_trade_summary()
        df = self. trade_tracker.to_dataframe()
        
        # Generate charts
        pnl_chart = self._generate_pnl_chart() if not df.empty else ""
        pair_performance_chart = self._generate_pair_performance_chart() if not df.empty else ""
        
        # Format date
        report_date = self.trade_tracker.date.strftime('%A, %B %d, %Y')
        
        # Create HTML content
        if df.empty:
            # No trades template
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                    h1, h2 {{ color: #0056b3; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    .header {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 25px; }}
                    .summary-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                    .no-trades {{ background-color: #fff8e1; padding: 20px; border-radius: 5px; border-left: 5px solid #ffc107; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Trading Report - {report_date}</h1>
                    </div>
                    
                    <div class="section">
                        <h2>Trading Summary</h2>
                        <div class="no-trades">
                            <h3>No Trades Executed</h3>
                            <p>No trades were completed during this trading session. This could be due to:</p>
                            <ul>
                                <li>Market conditions not meeting entry criteria</li>
                                <li>Trading hours restrictions</li>
                                <li>System maintenance</li>
                            </ul>
                            <p>The bot continues to monitor the market for trading opportunities.</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
        else:
            # Template with trades
            # ... (keep your existing HTML template for when there are trades)
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                    h1, h2 {{ color: #0056b3; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 25px; }}
                    .summary-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px; }}
                    th {{ 
                        background-color: #0056b3; 
                        color: white; 
                        padding: 12px 8px; 
                        text-align: left; 
                        white-space: nowrap; 
                    }}
                    td {{ 
                        padding: 10px 8px; 
                        border-bottom: 1px solid #ddd; 
                        white-space: nowrap; 
                    }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .positive {{ color: #28a745; }}
                    .negative {{ color: #dc3545; }}
                    .charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                    .chart {{ flex: 1; min-width: 300px; background-color: white; border-radius: 5px; padding: 10px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Trading Report - {report_date}</h1>
                    </div>
                    
                    <div class="section">
                        <h2>Trading Summary</h2>
                        <div class="summary-box">
                            <p><strong>Total Trades:</strong> {summary['total_trades']}</p>
                            <p><strong>Winning Trades:</strong> {summary['winning_trades']} ({summary['win_rate']*100:.1f}%)</p>
                            <p><strong>Losing Trades:</strong> {summary['losing_trades']}</p>
                            <p><strong>Total P&L:</strong> <span class="{'positive' if summary['total_pnl'] >= 0 else 'negative'}">${summary['total_pnl']:.2f}</span></p>
                            <p><strong>Average P&L per Trade:</strong> <span class="{'positive' if summary['average_pnl'] >= 0 else 'negative'}">${summary['average_pnl']:.2f}</span></p>
                            <p><strong>Best Trade:</strong> ${summary['best_trade_pnl']:.2f}</p>
                            <p><strong>Worst Trade:</strong> ${summary['worst_trade_pnl']:.2f}</p>
                            <p><strong>Best Performing Pair:</strong> {summary['best_pair']}</p>
                            <p><strong>Worst Performing Pair:</strong> {summary['worst_pair']}</p>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Performance Charts</h2>
                        <div class="charts">
                            <div class="chart">
                                <h3>P&L by Trade</h3>
                                {pnl_chart}
                            </div>
                            <div class="chart">
                                <h3>P&L by Trading Pair</h3>
                                {pair_performance_chart}
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Trade Details</h2>
                        {df.to_html(classes='table', index=False) if not df.empty else '<p>No trades to display</p>'}
                    </div>
                </div>
            </body>
            </html>
            """
        
        return html_content

        
    def _generate_pnl_chart(self) -> str:
        """Generate a base64 encoded image of the P&L chart"""
        try:
            df = self.trade_tracker.to_dataframe()
            
            if df.empty:
                return ""
            
            # Convert string PnL values back to float for plotting
            df['pnl_numeric'] = df['pnl'].str.replace('$', '').str.replace(',', '').astype(float)
            
            plt.figure(figsize=(8, 4))
            colors = ['#28a745' if x > 0 else '#dc3545' for x in df['pnl_numeric']]
            plt.bar(range(len(df)), df['pnl_numeric'], color=colors)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('P&L for Each Trade')
            plt.ylabel('Profit/Loss ($)')
            plt.tight_layout()
            
            # Save plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Encode the image to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return f'<img src="data:image/png;base64,{img_str}" style="width:100%;">'
        except Exception as e:
            return f"<p>Error generating P&L chart: {str(e)}</p>"
            
    def _generate_pair_performance_chart(self) -> str:
        """Generate a base64 encoded image of the pair performance chart"""
        try:
            df = self.trade_tracker.to_dataframe()
            
            if df.empty:
                return ""
                
            # Convert string PnL values back to float for aggregation
            df['pnl_numeric'] = df['pnl'].str.replace('$', '').str.replace(',', '').astype(float)
            
            # Group by trading pair and sum PnL
            pair_pnl = df.groupby('trading_pair')['pnl_numeric'].sum().sort_values()
            
            # Create bar chart
            plt.figure(figsize=(8, 4))
            colors = ['#28a745' if x > 0 else '#dc3545' for x in pair_pnl.values]
            pair_pnl.plot(kind='bar', color=colors)
            plt.title('P&L by Trading Pair')
            plt.ylabel('Profit/Loss ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Encode the image to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return f'<img src="data:image/png;base64,{img_str}" style="width:100%;">'
        except Exception as e:
            return f"<p>Error generating pair performance chart: {str(e)}</p>"
