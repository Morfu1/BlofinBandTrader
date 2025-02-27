import json
import logging
import asyncio
import websockets
import hmac
import hashlib
import base64
import time
from datetime import datetime
from uuid import uuid4
from typing import Callable, Dict, Optional
import random

class WebSocketManager:

    def __init__(self, config):
        self.config = config
        self.ws_public: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_private: Optional[websockets.WebSocketClientProtocol] = None
        self.callbacks: Dict[str, Callable] = {}
        self.last_ping_time = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.connected = False
        self.reconnect_attempts = 0
        self.MAX_RECONNECT_ATTEMPTS = 5
        self.last_candle_ts = 0  # Track last processed candle
        # Log configured trading pair and timeframe
        self.logger.info(
            f"WebSocket Manager initialized for trading pair: {self.config.TRADING_PAIR} with timeframe: {self.config.TIMEFRAME}"
        )

    async def generate_auth_payload(self) -> dict:
        """Generate authentication payload for WebSocket"""
        timestamp = str(int(time.time() * 1000))
        nonce = str(uuid4())

        # Fixed components for WebSocket auth
        method = "GET"
        path = "/users/self/verify"
        msg = f"{path}{method}{timestamp}{nonce}"

        # Generate signature
        signature = hmac.new(self.config.API_SECRET.encode(), msg.encode(),
                             hashlib.sha256).hexdigest()

        signature_b64 = base64.b64encode(signature.encode()).decode()

        self.logger.info("Generated WebSocket authentication payload")
        return {
            "op":
            "login",
            "args": [{
                "apiKey": self.config.API_KEY,
                "passphrase": self.config.API_PASSPHRASE,
                "timestamp": timestamp,
                "sign": signature_b64,
                "nonce": nonce
            }]
        }

    async def connect(self):
        """Establish WebSocket connections"""
        try:
            if self.connected:
                return

            # Connect to public WebSocket
            self.logger.info("Connecting to public WebSocket...")
            self.ws_public = await websockets.connect(
                self.config.WS_PUBLIC_URL,
                ping_interval=15,  # Reduced ping interval
                ping_timeout=10  # Shorter ping timeout
            )
            self.logger.info("Connected to public WebSocket")

            # Connect to private WebSocket and authenticate
            self.logger.info("Connecting to private WebSocket...")
            self.ws_private = await websockets.connect(
                self.config.WS_PRIVATE_URL, ping_interval=15, ping_timeout=10)
            auth_payload = await self.generate_auth_payload()
            await self.ws_private.send(json.dumps(auth_payload))

            # Wait for auth response
            response = await self.ws_private.recv()
            self.logger.info(f"Authentication response: {response}")

            # Subscribe to required channels
            await self.subscribe_channels()
            self.connected = True
            self.reconnect_attempts = 0
            self.logger.info("WebSocket connections established successfully")

        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            self.connected = False
            await self.reconnect()

    async def subscribe_channels(self):
        """Subscribe to required WebSocket channels"""
        try:
            # Get all trading pairs from config
            trading_pairs = (self.config.MULTIPLE_COINS
                             if self.config.COIN_SELECTION_MODE == "multiple"
                             else [self.config.SINGLE_COIN])
            
            self.logger.info(f"Subscribing to channels for pairs: {trading_pairs}")
            
            # Subscribe to candlestick channel for each trading pair
            for pair in trading_pairs:
                candle_sub = {
                    "op": "subscribe",
                    "args": [{
                        "channel": f"candle{self.config.TIMEFRAME}",
                        "instId": pair
                    }]
                }
                
                self.logger.info(f"Subscribing to candlestick channel for {pair} with timeframe {self.config.TIMEFRAME}")
                await self.ws_public.send(json.dumps(candle_sub))
                
                # Add a small delay between subscriptions to ensure proper order
                await asyncio.sleep(0.5)
            
            # Wait for all subscription responses
            for _ in trading_pairs:
                response = await self.ws_public.recv()
                self.logger.info(f"Subscription response received: {response}")
            
        except Exception as e:
            self.logger.error(f"Subscription error: {str(e)}")
            self.connected = False
            await self.reconnect()

    async def heartbeat(self):
        """Send heartbeat to keep connection alive"""
        while True:
            try:
                if self.connected:
                    if time.time(
                    ) - self.last_ping_time > 15:  # Reduced interval to 15 seconds
                        if self.ws_public and self.ws_private:
                            await self.ws_public.ping()
                            await self.ws_private.ping()
                            self.last_ping_time = time.time()
                            self.logger.debug("Heartbeat sent successfully")
                    await asyncio.sleep(5)  # More frequent checks
                else:
                    await self.reconnect()
                    await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {str(e)}")
                self.connected = False
                await self.reconnect()

    async def reconnect(self):
        """Reconnect WebSocket on connection loss with enhanced error handling and recovery"""
        while True:  # Keep trying to reconnect indefinitely
            try:
                if self.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                    self.logger.warning(
                        f"Maximum reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached. "
                        "Entering extended recovery mode..."
                    )
                    # Extended recovery mode
                    await asyncio.sleep(60)  # 1 minute cooldown
                    self.reconnect_attempts = 0
                    self.logger.info("Reset reconnection attempts counter after cooldown")
                
                self.reconnect_attempts += 1
                
                # Calculate backoff time with jitter to prevent thundering herd
                base_wait = min(5 * self.reconnect_attempts, 30)
                jitter = random.uniform(0, 2)  # Add up to 2 seconds of random jitter
                wait_time = base_wait + jitter
                
                self.logger.info(
                    f"Attempting to reconnect (attempt {self.reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS}) "
                    f"after {wait_time:.2f}s delay..."
                )
                
                # Wait before attempting reconnection
                await asyncio.sleep(wait_time)
                
                # Close existing connections if any
                if self.ws_public:
                    await self.ws_public.close()
                if self.ws_private:
                    await self.ws_private.close()
                
                # Attempt to reconnect
                await self.connect()
                
                # If connection successful, break the loop
                if self.connected:
                    self.logger.info("Reconnection successful!")
                    self.reconnect_attempts = 0  # Reset counter on successful connection
                    break
                    
            except websockets.exceptions.InvalidStatusCode as e:
                self.logger.error(f"Invalid status code during reconnection: {e}")
                self.connected = False
                # Don't increment retry counter for authentication issues
                continue
                
            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                self.logger.error(f"Connection closed during reconnection attempt: {e}")
                self.connected = False
                continue
                
            except asyncio.CancelledError:
                self.logger.warning("Reconnection attempt cancelled")
                raise
                
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during reconnection (attempt {self.reconnect_attempts}): {str(e)}"
                )
                self.connected = False
                
                # If this is a critical error, maybe we should break
                if isinstance(e, (ConnectionRefusedError, OSError)):
                    self.logger.critical("Critical connection error - manual intervention may be required")
                    raise

    async def message_handler(self):
        """Handle incoming WebSocket messages"""
        while True:
            try:
                if not self.connected:
                    await asyncio.sleep(1)
                    continue
                
                # Handle public messages
                message = await self.ws_public.recv()
                if message == 'pong':
                    self.logger.debug("Received pong response")
                    continue
                
                data = json.loads(message)
                
                # Skip subscription confirmations in normal handler flow
                if 'event' in data and data['event'] == 'subscribe':
                    self.logger.debug(f"Received subscription confirmation: {data}")
                    continue
                
                if 'event' in data and data['event'] == 'error':
                    self.logger.error(f"WebSocket error: {data}")
                    continue
                
                if 'data' in data and 'arg' in data:
                    channel = data['arg'].get('channel', '')
                    inst_id = data['arg'].get('instId', '')
                    
                    # Get all trading pairs from config
                    trading_pairs = (self.config.MULTIPLE_COINS
                                    if self.config.COIN_SELECTION_MODE == "multiple"
                                    else [self.config.SINGLE_COIN])
                    
                    # Ensure we have the trading pair in the instance ID
                    if not inst_id or inst_id not in trading_pairs:
                        self.logger.warning(f"Received data for unknown trading pair: {inst_id}")
                        continue
                    
                    # Process candle data
                    if channel.startswith('candle') and self.callbacks.get('candle'):
                        candle_data = data['data']
                        
                        # Validate candle data structure
                        if not candle_data or not isinstance(candle_data, list):
                            self.logger.error(f"Invalid candle data structure: {candle_data}")
                            continue
                        
                        try:
                            # Pass the trading pair along with the data to the callback
                            await self.callbacks['candle'](candle_data, inst_id)
                            self.logger.debug(f"Successfully processed candle update for {inst_id}")
                        except Exception as e:
                            self.logger.error(f"Error processing candle data for {inst_id}: {str(e)}")
                            continue
                        
            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                self.logger.error(f"Connection closed: {str(e)}")
                self.connected = False
                await self.reconnect()
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Message handling error: {str(e)}")
                self.connected = False
                await self.reconnect()

    def register_callback(self, channel: str, callback: Callable):
        """Register callback for specific channel"""
        self.callbacks[channel] = callback
        self.logger.info(f"Registered callback for channel: {channel}")

    async def close(self):
        """Close WebSocket connections"""
        self.logger.info("Closing WebSocket connections...")
        self.connected = False
        if self.ws_public:
            await self.ws_public.close()
        if self.ws_private:
            await self.ws_private.close()
        self.logger.info("WebSocket connections closed")
