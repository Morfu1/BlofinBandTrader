import logging
import hmac
import hashlib
import base64
import time
from datetime import datetime
from uuid import uuid4
from typing import Dict, Optional

class Utils:
    @staticmethod
    def setup_logging():
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def generate_signature(secret: str, method: str, path: str, 
                          timestamp: str, nonce: str, body: Optional[Dict] = None) -> str:
        """Generate API request signature"""
        body_str = ""
        if body:
            import json
            body_str = json.dumps(body)
            
        message = f"{path}{method}{timestamp}{nonce}{body_str}"
        
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return base64.b64encode(signature.encode()).decode()

    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp in milliseconds"""
        return str(int(time.time() * 1000))

    @staticmethod
    def get_nonce() -> str:
        """Generate unique nonce"""
        return str(uuid4())
