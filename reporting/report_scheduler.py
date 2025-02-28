import pytz
from datetime import datetime, time, timedelta
import logging
import schedule
import asyncio

class ReportScheduler:
    def __init__(self, trading_bot, report_service):
        """Initialize scheduler with trading bot and report service"""
        self.trading_bot = trading_bot
        self.report_service = report_service
        self.logger = logging.getLogger(__name__)
        
        # Define market close time in EST and convert to local time
        est_tz = pytz.timezone('US/Eastern')
        local_tz = pytz.timezone('Europe/Bucharest')
        
        # Create EST market close time (16:24 EST)
        est_close = datetime.now(est_tz).replace(hour=16, minute=4, second=0, microsecond=0)
        
        # Convert to local time
        local_time = est_close.astimezone(local_tz)
        self.report_time = local_time.strftime("%H:%M")
        
        self.logger.info(
            f"Daily reports scheduled for {self.report_time} local time (Romania) "
            f"({est_close.strftime('%H:%M')} EST)"
        )
    
    async def schedule_daily_report(self):
        """Schedule daily reports"""
        try:
            # Schedule for every day
            schedule.every().day.at(self.report_time).do(
                lambda: asyncio.create_task(self.generate_and_send_report())
            )
            
            self.logger.info(f"Report scheduler initialized. Reports will be sent daily at {self.report_time} local time")
            
            last_log_time = None
            while True:
                now = datetime.now()
                next_run = schedule.next_run()
                
                # Log status only once per hour
                current_hour = now.replace(minute=0, second=0, microsecond=0)
                if last_log_time is None or current_hour > last_log_time:
                    if next_run:
                        time_until_next = next_run - now
                        self.logger.info(
                            f"Next report scheduled for: {self.report_time} "
                            f"(in {time_until_next})"
                        )
                    last_log_time = current_hour
                
                # Run any pending tasks
                schedule.run_pending()
                
                # Sleep for 30 seconds instead of 60 to be more responsive
                await asyncio.sleep(30)
                
        except Exception as e:
            self.logger.error(f"Error in report scheduler: {str(e)}")
            self.logger.exception("Full traceback:")
            # Re-raise to ensure the error is handled by the main loop
            raise
    
    async def generate_and_send_report(self):
        """Generate and send the daily trading report"""
        self.logger.info("Generating end-of-day trading report...")
        try:
            # Fetch closed trades from exchange for the past 24 hours
            start_time = datetime.now() - timedelta(hours=24)
            self.logger.info(f"Fetching trades from {start_time} to {datetime.now()}")
            await self.trading_bot.get_closed_trades_from_exchange(
                start_time=start_time,
                end_time=datetime.now()
            )
            
            # Make sure trade tracker has the latest data
            if hasattr(self.trading_bot, 'trade_tracker'):
                trade_count = len(self.trading_bot.trade_tracker.trades)
                self.logger.info(f"Found {trade_count} trades in the last 24 hours")
                
                # Update report service with latest trade data
                self.report_service.set_trade_tracker(self.trading_bot.trade_tracker)
                
                # Generate custom message with key stats
                if trade_count > 0:
                    summary = self.trading_bot.trade_tracker.get_trade_summary()
                    additional_message = (
                        f"Trading Day Summary - {datetime.now().strftime('%A, %B %d, %Y')}\n\n"
                        f"Total Trades: {summary['total_trades']}\n"
                        f"Winning Trades: {summary['winning_trades']}\n"
                        f"Losing Trades: {summary['losing_trades']}\n"
                        f"Win Rate: {summary['win_rate']*100:.1f}%\n"
                        f"Total P&L: ${summary['total_pnl']:.2f}\n"
                        f"Best Performing Pair: {summary['best_pair']}\n"
                        f"Worst Performing Pair: {summary['worst_pair']}\n\n"
                        f"See attached HTML report for complete details."
                    )
                else:
                    additional_message = (
                        f"Trading Day Summary - {datetime.now().strftime('%A, %B %d, %Y')}\n\n"
                        "No trades were executed in the last 24 hours.\n"
                        "This could be due to:\n"
                        "- Market conditions not meeting entry criteria\n"
                        "- Trading hours restrictions\n"
                        "- System maintenance\n\n"
                        "The bot continues to monitor the market for trading opportunities."
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
            self.logger.exception("Full traceback:")

