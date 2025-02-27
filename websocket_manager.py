import websockets
import packaging.version

# Detect WebSockets version for compatibility
try:
    WEBSOCKETS_VERSION = packaging.version.parse(websockets.__version__)
    IS_WEBSOCKETS_GTE_15 = WEBSOCKETS_VERSION >= packaging.version.parse("15.0")
except (AttributeError, ValueError):
    # Default to assuming newer version if detection fails
    IS_WEBSOCKETS_GTE_15 = True

import json
import logging
import asyncio
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
                ping_interval=None,  # Disable automatic pings, we'll handle it
                ping_timeout=20,     # Increased timeout
                close_timeout=10     # Add explicit close timeout
            )
            self.logger.info("Connected to public WebSocket")

            # Connect to private WebSocket and authenticate
            self.logger.info("Connecting to private WebSocket...")
            self.ws_private = await websockets.connect(
                self.config.WS_PRIVATE_URL,
                ping_interval=None,  # Disable automatic pings, we'll handle it
                ping_timeout=20,     # Increased timeout
                close_timeout=10     # Add explicit close timeout
            )
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
        """Subscribe to required WebSocket channels with improved efficiency"""
        try:
            # Get all trading pairs from config
            trading_pairs = (
                self.config.MULTIPLE_COINS
                if self.config.COIN_SELECTION_MODE == "multiple"
                else [self.config.SINGLE_COIN]
            )
            
            self.logger.info(f"Subscribing to channels for {len(trading_pairs)} pairs")
            
            # Batch subscriptions into groups of 5 to avoid overwhelming the connection
            batch_size = 5
            for i in range(0, len(trading_pairs), batch_size):
                batch = trading_pairs[i:i+batch_size]
                subscription_tasks = []
                
                for pair in batch:
                    candle_sub = {
                        "op": "subscribe",
                        "args": [{
                            "channel": f"candle{self.config.TIMEFRAME}",
                            "instId": pair
                        }]
                    }
                    self.logger.info(f"Subscribing to candlestick channel for {pair}")
                    subscription_tasks.append(self.ws_public.send(json.dumps(candle_sub)))
                
                # Send batch of subscriptions in parallel
                await asyncio.gather(*subscription_tasks)
                
                # Small delay between batches
                await asyncio.sleep(1)
                
            # Wait for subscription responses (with timeout protection)
            response_count = 0
            expected_responses = len(trading_pairs)
            
            # Set a timeout for receiving all responses
            start_time = time.time()
            timeout = 30  # 30 seconds timeout
            
            while response_count < expected_responses and (time.time() - start_time < timeout):
                try:
                    response = await asyncio.wait_for(self.ws_public.recv(), timeout=5)
                    data = json.loads(response)
                    if 'event' in data and data['event'] == 'subscribe':
                        response_count += 1
                        self.logger.debug(f"Subscription response {response_count}/{expected_responses}")
                
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for subscription response")
                    break
            
            self.logger.info(f"Completed {response_count}/{expected_responses} subscriptions")
            
        except Exception as e:
            self.logger.error(f"Subscription error: {str(e)}")
            self.connected = False
            await self.reconnect()

    async def heartbeat(self):
        """Send heartbeat to keep connection alive"""
        while True:
            try:
                if self.connected and self.ws_public and self.ws_private:
                    # Use a more robust connection check approach
                    try:
                        # First try to send a ping to check if connections are alive
                        await asyncio.wait_for(self.ws_public.ping(), timeout=2)
                        await asyncio.wait_for(self.ws_private.ping(), timeout=2)
                        self.last_ping_time = time.time()
                        self.logger.debug("Heartbeat sent successfully")
                    except (asyncio.TimeoutError, ConnectionError, Exception) as e:
                        self.logger.warning(f"Connection issue detected during heartbeat: {str(e)}")
                        self.connected = False
                        await self.reconnect()
                elif not self.connected:
                    await asyncio.sleep(5)
                
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {str(e)}")
                self.connected = False
                await self.reconnect()

    async def reconnect(self):
        """Reconnect WebSocket with enhanced error handling"""
        while True:
            try:
                if self.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                    self.logger.warning(
                        f"Maximum reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached. "
                        "Entering extended recovery mode..."
                    )
                    await asyncio.sleep(60)  # 1 minute cooldown
                    self.reconnect_attempts = 0
                    self.logger.info("Reset reconnection attempts counter after cooldown")
                
                self.reconnect_attempts += 1
                
                # Calculate backoff time with jitter
                base_wait = min(5 * self.reconnect_attempts, 30)
                jitter = random.uniform(0, 2)
                wait_time = base_wait + jitter
                
                self.logger.info(
                    f"Attempting to reconnect (attempt {self.reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS}) "
                    f"after {wait_time:.2f}s delay..."
                )
                
                # Wait before attempting reconnection
                await asyncio.sleep(wait_time)
                
                # Properly close existing connections if any
                await self._safe_close_connections()
                
                # Attempt to reconnect
                await self.connect()
                
                # If connection successful, break the loop
                if self.connected:
                    self.logger.info("Reconnection successful!")
                    self.reconnect_attempts = 0
                    break
                
            except Exception as e:
                self.logger.error(f"Unexpected error during reconnection: {str(e)}")
                self.connected = False
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _safe_close_connections(self):
        """Safely close existing WebSocket connections"""
        try:
            if self.ws_public:
                try:
                    await self.ws_public.close(code=1000, reason="Reconnecting")
                except Exception as e:
                    self.logger.warning(f"Error closing public websocket: {str(e)}")
                self.ws_public = None
            
            if self.ws_private:
                try:
                    await self.ws_private.close(code=1000, reason="Reconnecting")
                except Exception as e:
                    self.logger.warning(f"Error closing private websocket: {str(e)}")
                self.ws_private = None
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")

    async def message_handler(self):
        """Handle incoming WebSocket messages"""
        while True:
            try:
                if not self.connected:
                    await asyncio.sleep(1)
                    continue

                # Handle public messages
                message = await self.ws_public.recv()
                
                # Update last message timestamp
                self.last_message_time = time.time()
                
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

    async def connection_health_monitor(self):
        """Monitor connection health and proactively reconnect if needed"""
        while True:
            try:
                if self.connected:
                    # Check connection health by trying to ping
                    connection_alive = True
                    try:
                        if self.ws_public:
                            await asyncio.wait_for(self.ws_public.ping(), timeout=2)
                        if self.ws_private:
                            await asyncio.wait_for(self.ws_private.ping(), timeout=2)
                    except Exception:
                        connection_alive = False
                    
                    if not connection_alive:
                        self.logger.warning("Detected connection issue during health check")
                        self.connected = False
                        await self.reconnect()
                    
                    # Check last message time
                    if hasattr(self, 'last_message_time') and time.time() - self.last_message_time > 60:
                        self.logger.warning("No messages received in 60 seconds, reconnecting")
                        self.connected = False
                        await self.reconnect()
            
                await asyncio.sleep(15)
            
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")
                await asyncio.sleep(15)
