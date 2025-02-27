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
        df = self.trade_tracker.to_dataframe()
        
        # Generate charts
        pnl_chart = self._generate_pnl_chart() if not df.empty else ""
        pair_performance_chart = self._generate_pair_performance_chart() if not df.empty else ""
        
        # Format date
        report_date = self.trade_tracker.date.strftime('%A, %B %d, %Y')
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Daily Trading Report - {report_date}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .summary-container {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    margin-bottom: 30px;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    width: 30%;
                }}
                .metric-title {{
                    font-size: 14px;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                }}
                .positive {{
                    color: #28a745;
                }}
                .negative {{
                    color: #dc3545;
                }}
                .neutral {{
                    color: #007bff;
                }}
                .chart-container {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}
                .chart {{
                    width: 48%;
                    background-color: #fff;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 25px 0;
                    font-size: 14px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th {{
                    background-color: #007bff;
                    color: #ffffff;
                    text-align: left;
                    padding: 12px;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #dddddd;
                }}
                tr:nth-of-type(even) {{
                    background-color: #f3f3f3;
                }}
                tr:last-of-type td {{
                    border-bottom: none;
                }}
                .section-title {{
                    margin-top: 40px;
                    margin-bottom: 20px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    font-size: 12px;
                    color: #777;
                }}
                .no-data {{
                    text-align: center;
                    padding: 40px;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Daily Trading Report</h1>
                <h2>{report_date}</h2>
            </div>
            
            <!-- Summary Metrics -->
            <h2 class="section-title">Daily Performance Summary</h2>
        """
            
        if summary["total_trades"] > 0:
            pnl_color = "positive" if summary["total_pnl"] > 0 else "negative"
            win_rate_color = "positive" if summary["win_rate"] >= 0.5 else "negative"
            
            html_content += f"""
            <div class="summary-container">
                <div class="summary-box">
                    <div class="metric-title">Total Trades</div>
                    <div class="metric-value neutral">{summary["total_trades"]}</div>
                </div>
                <div class="summary-box">
                    <div class="metric-title">Win Rate</div>
                    <div class="metric-value {win_rate_color}">{summary["win_rate"]*100:.1f}%</div>
                </div>
                <div class="summary-box">
                    <div class="metric-title">Total P&L</div>
                    <div class="metric-value {pnl_color}">${summary["total_pnl"]:.2f}</div>
                </div>
            </div>
            
            <div class="summary-container">
                <div class="summary-box">
                    <div class="metric-title">Winning Trades</div>
                    <div class="metric-value positive">{summary["winning_trades"]}</div>
                </div>
                <div class="summary-box">
                    <div class="metric-title">Losing Trades</div>
                    <div class="metric-value negative">{summary["losing_trades"]}</div>
                </div>
                <div class="summary-box">
                    <div class="metric-title">Average P&L</div>
                    <div class="metric-value {pnl_color}">${summary["average_pnl"]:.2f}</div>
                </div>
            </div>
            
            <!-- Performance Charts -->
            <div class="chart-container">
                <div class="chart">
                    <h3>P&L Distribution</h3>
                    {pnl_chart}
                </div>
                <div class="chart">
                    <h3>Trading Pair Performance</h3>
                    {pair_performance_chart}
                </div>
            </div>
            """
        else:
            html_content += """
            <div class="no-data">
                <p>No trades were closed today.</p>
            </div>
            """
            
        html_content += """
            <!-- Detailed Trade List -->
            <h2 class="section-title">Closed Trade Details</h2>
        """
        
        if not df.empty:
            # Create an HTML table from the dataframe
            trade_table = df[['trading_pair', 'position_type', 'entry_time', 'exit_time', 
                              'entry_price', 'exit_price', 'take_profit', 'stop_loss', 
                              'pnl', 'pnl_percent', 'exit_reason']].to_html(
                index=False, 
                classes='trade-table',
                formatters={
                    'entry_time': lambda x: x.strftime('%H:%M:%S'),
                    'exit_time': lambda x: x.strftime('%H:%M:%S')
                }
            )
            html_content += trade_table
        else:
            html_content += """
            <div class="no-data">
                <p>No trades were closed today.</p>
            </div>
            """
            
        html_content += """
            <!-- Footer -->
            <div class="footer">
                <p>This report was automatically generated by the Trading Bot.</p>
                <p>© 2025 Trading Bot System</p>
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
