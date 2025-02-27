import asyncio
import aioschedule as schedule
from datetime import datetime, time, timedelta
import pytz
import logging

class ReportScheduler:
    def __init__(self, trading_bot, report_service):
        """Initialize scheduler with trading bot and report service"""
        self.trading_bot = trading_bot
        self.report_service = report_service
        self.logger = logging.getLogger(__name__)
        
        # Define market close time (e.g., 16:00 EST for US markets)
        # Adjust this based on the trading hours of your market
        self.market_close_time = time(16, 0)  # 4:00 PM
        self.timezone = pytz.timezone('US/Eastern')  # Adjust timezone as needed
    
    async def schedule_daily_report(self):
        """Schedule daily reports at market close"""
        # Schedule report generation 5 minutes after market close
        report_time = (datetime.combine(datetime.today(), self.market_close_time) + 
                      timedelta(minutes=5)).time()
        
        schedule.every().monday.at(report_time.strftime("%H:%M")).do(self.generate_and_send_report)
        schedule.every().tuesday.at(report_time.strftime("%H:%M")).do(self.generate_and_send_report)
        schedule.every().wednesday.at(report_time.strftime("%H:%M")).do(self.generate_and_send_report)
        schedule.every().thursday.at(report_time.strftime("%H:%M")).do(self.generate_and_send_report)
        schedule.every().friday.at(report_time.strftime("%H:%M")).do(self.generate_and_send_report)
        
        self.logger.info(f"Daily reports scheduled for {report_time.strftime('%H:%M')} EST")
        
        # Run the scheduler
        while True:
            await schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
    
    async def generate_and_send_report(self):
        """Generate and send the daily trading report"""
        self.logger.info("Generating end-of-day trading report...")
        
        try:
            # Make sure trade tracker has the latest data
            if hasattr(self.trading_bot, 'trade_tracker'):
                # Update report service with latest trade data
                self.report_service.set_trade_tracker(self.trading_bot.trade_tracker)
                
                # Generate custom message with key stats
                summary = self.trading_bot.trade_tracker.get_trade_summary()
                
                additional_message = (
                    f"Trading Day Summary - {datetime.now().strftime('%A, %B %d, %Y')}\n\n"
                    f"Total Trades: {summary['total_trades']}\n"
                    f"Winning Trades: {summary['winning_trades']}\n"
                    f"Losing Trades: {summary['losing_trades']}\n"
                    f"Win Rate: {summary['win_rate']*100:.1f}%\n"
                    f"Total P&L: ${summary['total_pnl']:.2f}\n\n"
                    f"See attached HTML report for complete details."
                )
                
                # Send report
                success = self.report_service.send_daily_report(additional_message)
                
                if success:
                    self.logger.info("End-of-day report sent successfully")
                else:
                    self.logger.error("Failed to send end-of-day report")
                
            else:
                self.logger.error("Trading bot has no trade tracker")
                
        except Exception as e:
            self.logger.error(f"Error generating end-of-day report: {str(e)}")
