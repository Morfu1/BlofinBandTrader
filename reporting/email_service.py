import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from .report_generator import TradingReportGenerator

class TradingReportEmailService:
    def __init__(self, trade_tracker=None):
        """Initialize email service using credentials from .env file"""
        # Load environment variables
        load_dotenv()
        
        # Email configuration
        self.gmail_user = os.getenv("GMAIL_USER")
        self.gmail_password = os.getenv("GMAIL_PASSWORD")
        self.notification_email = os.getenv("NOTIFICATION_EMAIL")
        
        # Validate email configuration
        if not all([self.gmail_user, self.gmail_password, self.notification_email]):
            raise ValueError("Email credentials not properly configured in .env file")
        
        self.logger = logging.getLogger(__name__)
        self.trade_tracker = trade_tracker
        self.report_generator = TradingReportGenerator(trade_tracker) if trade_tracker else None
    
    def set_trade_tracker(self, trade_tracker):
        """Set the trade tracker after initialization"""
        self.trade_tracker = trade_tracker
        self.report_generator = TradingReportGenerator(trade_tracker)
    
    def send_daily_report(self, additional_message: str = "") -> bool:
        """Generate and send daily trading report via email"""
        if not self.trade_tracker or not self.report_generator:
            self.logger.error("Cannot send report: Trade Tracker not initialized")
            return False
            
        try:
            # Generate HTML report
            html_content = self.report_generator.generate_html_report()
            
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Daily Trading Report - {datetime.now().strftime('%A, %B %d, %Y')}"
            msg['From'] = self.gmail_user
            msg['To'] = self.notification_email
            
            # Add custom message if provided
            if additional_message:
                text_part = MIMEText(additional_message, 'plain')
                msg.attach(text_part)
            
            # Attach HTML report
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Create secure SSL context
            context = ssl.create_default_context()
            
            # Send email
            self.logger.info("Sending daily trading report email...")
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
                server.login(self.gmail_user, self.gmail_password)
                server.sendmail(self.gmail_user, self.notification_email, msg.as_string())
            
            self.logger.info(f"Daily trading report sent successfully to {self.notification_email}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error sending daily trading report: {str(e)}")
            return False
