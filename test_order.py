import asyncio
import logging
from datetime import datetime
from config import Config
from risk_manager import RiskManager
from utils import Utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def close_existing_positions(risk_manager):
    """Close any existing positions before testing"""
    try:
        current_positions = await risk_manager.get_current_positions()

        if current_positions:
            logger.info("Found existing position - attempting to close...")

            # Get current position details
            position_size = current_positions['positions']
            margin_mode = current_positions['marginMode']
            leverage = current_positions['leverage']

            # Determine close side based on position direction
            close_side = "sell" if float(position_size) > 0 else "buy"

            close_order = {
                "instId": risk_manager.config.TRADING_PAIR,
                "marginMode": margin_mode,
                "side": close_side,
                "orderType": "market",
                "size": abs(float(position_size)),  # Use absolute value for size
                "lever": leverage,
                "reduceOnly": "true"  # Ensure we only close existing position
            }

            logger.info(f"\nClosing Position:\n"
                       f"========================================\n"
                       f"Current Size: {position_size}\n"
                       f"Side: {close_side}\n"
                       f"Order: {close_order}\n"
                       f"========================================")

            close_response = await place_test_order(close_order, risk_manager.config)

            if close_response and close_response.get('code') == '0':
                logger.info("Successfully sent close order")
                # Wait a moment for position to close
                await asyncio.sleep(2)

                # Verify position is closed
                verify_position = await risk_manager.get_current_positions()
                if verify_position:
                    logger.error("Position still open after close attempt")
                    return False
                logger.info("Position successfully closed")
                return True
            else:
                logger.error(f"Failed to close position: {close_response}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error closing positions: {str(e)}")
        return False

async def test_order_placement():
    """Test order placement with Stop Loss and Take Profit"""
    try:
        config = Config()
        risk_manager = RiskManager(config=config)

        # Initialize risk manager with trading parameters
        await risk_manager.initialize()

        # Close any existing positions first
        logger.info("\nChecking and closing any existing positions...")
        if not await close_existing_positions(risk_manager):
            logger.error("Failed to close existing positions - aborting test")
            return

        # Double check positions are closed
        initial_positions = await risk_manager.get_current_positions()
        if initial_positions:
            logger.error("Still have existing positions after closing attempt - aborting test")
            return

        # Test parameters
        entry_price = 0.002911  # Current market price from logs
        position_type = "long"  # Test long position

        logger.info(f"\nStarting Order Placement Test:\n"
                   f"========================================\n"
                   f"Trading Pair: {config.TRADING_PAIR}\n"
                   f"Position Type: {position_type}\n"
                   f"Entry Price: ${entry_price}\n"
                   f"========================================")

        # Calculate position size based on USDT margin
        position_info = risk_manager.calculate_position_size(entry_price)

        if not position_info:
            logger.error("Failed to calculate position size")
            return

        # Calculate SL/TP levels
        stop_loss = await risk_manager.calculate_stop_loss(entry_price, position_type)
        take_profit = risk_manager.calculate_take_profit(entry_price, stop_loss, position_type)

        # Market order for entry
        entry_order = {
            "instId": config.TRADING_PAIR,
            "marginMode": position_info["marginMode"],
            "side": "buy" if position_type == "long" else "sell",
            "orderType": "market",
            "size": position_info["size"],
            "lever": position_info["leverage"]
        }

        logger.info(f"\nEntry Order Details:\n"
                   f"========================================\n"
                   f"{entry_order}\n"
                   f"========================================")

        # Place entry order
        entry_order_response = await place_test_order(entry_order, config)

        if not entry_order_response or entry_order_response.get('code') != '0':
            logger.error(f"Entry order failed: {entry_order_response}")
            return

        # Verify position was opened
        logger.info("\nChecking positions after order placement...")
        current_positions = await risk_manager.get_current_positions()

        if not current_positions:
            logger.error("Position not found after order placement")
            return

        # Place TPSL order using updated format
        tpsl_order = {
            "instId": config.TRADING_PAIR,
            "marginMode": position_info["marginMode"],
            "side": "sell" if position_type == "long" else "buy",
            "tpTriggerPrice": str(take_profit),
            "tpOrderPrice": str(take_profit),  # Execute at trigger price
            "slTriggerPrice": str(stop_loss),
            "slOrderPrice": str(stop_loss),  # Execute at trigger price
            "size": position_info["size"],
            "tpTriggerPxType": "mark",  # Use mark price for trigger
            "slTriggerPxType": "mark"   # Use mark price for trigger
        }

        logger.info(f"\nTP/SL Order Details:\n"
                   f"========================================\n"
                   f"{tpsl_order}\n"
                   f"========================================")

        # Place TPSL order
        tpsl_response = await place_test_order(tpsl_order, config, "/api/v1/trade/order-tpsl")

        if not tpsl_response or tpsl_response.get('code') != '0':
            logger.error(f"TPSL order failed: {tpsl_response}")

        logger.info(f"\nTest Results:\n"
                   f"========================================\n"
                   f"Entry Order Status: {'Success' if entry_order_response.get('code') == '0' else 'Failed'}\n"
                   f"Position Status: {'Opened' if current_positions else 'Failed to Open'}\n"
                   f"TPSL Order Status: {'Success' if tpsl_response and tpsl_response.get('code') == '0' else 'Failed'}\n"
                   f"Position Size: {position_info['size']}\n"
                   f"Take Profit: ${take_profit:.6f}\n"
                   f"Stop Loss: ${stop_loss:.6f}\n"
                   f"========================================")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.exception("Full traceback:")

async def place_test_order(order, config, endpoint="/api/v1/trade/order"):
    """Place test order via REST API"""
    try:
        import aiohttp

        url = f"{config.REST_URL}{endpoint}"
        timestamp = Utils.get_timestamp()
        nonce = Utils.get_nonce()

        # Generate signature
        signature = Utils.generate_signature(
            config.API_SECRET,
            "POST",
            endpoint,
            timestamp,
            nonce,
            order
        )

        headers = {
            "ACCESS-KEY": config.API_KEY,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-NONCE": nonce,
            "ACCESS-PASSPHRASE": config.API_PASSPHRASE,
            "Content-Type": "application/json"
        }

        logger.debug(f"Sending order request: {order}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=order, headers=headers) as response:
                data = await response.json()
                return data

    except Exception as e:
        logger.error(f"Error placing test order: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting order placement test")
    asyncio.run(test_order_placement())