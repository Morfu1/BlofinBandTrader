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
        self.connection_lifespan = 3600  # 1 hour in seconds
        self.connection_start_time = 0
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
            # Add connection lock to prevent multiple simultaneous connections
            if hasattr(self, '_connecting') and self._connecting:
                return
            self._connecting = True

            if self.connected:
                self._connecting = False
                return

            # Close existing connections if any
            await self.close()

            # Connect to public WebSocket
            self.logger.info("Connecting to public WebSocket...")
            self.ws_public = await websockets.connect(
                self.config.WS_PUBLIC_URL,
                ping_interval=30,  # Increased from 20
                ping_timeout=30,   # Increased from 20
                close_timeout=15   # Increased from 10
            )
            self.logger.info("Connected to public WebSocket")

            # Connect to private WebSocket and authenticate
            self.logger.info("Connecting to private WebSocket...")
            self.ws_private = await websockets.connect(
                self.config.WS_PRIVATE_URL,
                ping_interval=30,  # Increased from 20
                ping_timeout=30,   # Increased from 20
                close_timeout=15   # Increased from 10
            )
            auth_payload = await self.generate_auth_payload()
            await self.ws_private.send(json.dumps(auth_payload))

            # Wait for auth response with timeout
            try:
                response = await asyncio.wait_for(self.ws_private.recv(), timeout=10)
                self.logger.info(f"Authentication response: {response}")
                
                # Verify authentication success
                resp_data = json.loads(response)
                if resp_data.get('code') != '0':
                    raise Exception(f"Authentication failed: {response}")
            except asyncio.TimeoutError:
                raise Exception("Authentication response timeout")

            # Subscribe to required channels
            await self.subscribe_channels()
            self.connected = True
            self.reconnect_attempts = 0
            self.connection_start_time = time.time()  # Record connection start time
            self.logger.info("WebSocket connections established successfully")

        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            self.connected = False
            await self.reconnect()
        finally:
            self._connecting = False

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
            
            # Create a single subscription message for all pairs
            subscription_args = []
            for pair in trading_pairs:
                subscription_args.append({
                    "channel": f"candle{self.config.TIMEFRAME}",
                    "instId": pair
                })
            
            subscription_message = {
                "op": "subscribe",
                "args": subscription_args
            }
            
            # Send single subscription message
            await self.ws_public.send(json.dumps(subscription_message))
            
            # Wait for subscription confirmations with timeout
            timeout = 30  # 30 seconds total timeout
            start_time = time.time()
            confirmed_pairs = set()
            
            while len(confirmed_pairs) < len(trading_pairs):
                if time.time() - start_time > timeout:
                    raise Exception("Subscription confirmation timeout")
                    
                try:
                    response = await asyncio.wait_for(self.ws_public.recv(), timeout=5)
                    data = json.loads(response)
                    if data.get('event') == 'subscribe' and data.get('arg', {}).get('instId'):
                        confirmed_pairs.add(data['arg']['instId'])
                except asyncio.TimeoutError:
                    continue
                
            self.logger.info(f"Completed {len(confirmed_pairs)}/{len(trading_pairs)} subscriptions")
            
        except Exception as e:
            self.logger.error(f"Subscription error: {str(e)}")
            raise

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
                    del data  # Help with memory management
                    continue
                
                if 'event' in data and data['event'] == 'error':
                    self.logger.error(f"WebSocket error: {data}")
                    del data  # Help with memory management
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
                        
                # Explicitly clean up large data objects
                del data
                        
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
                    
                    # Check connection lifespan - proactively reconnect after the configured lifespan
                    if self.connection_start_time > 0 and time.time() - self.connection_start_time > self.connection_lifespan:
                        self.logger.info(f"Connection lifespan of {self.connection_lifespan}s reached, performing scheduled reconnection")
                        self.connected = False
                        await self.reconnect()
            
                await asyncio.sleep(15)
            
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")
                await asyncio.sleep(15)
