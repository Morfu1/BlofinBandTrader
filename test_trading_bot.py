import asyncio
import pandas as pd
import json
from trading_bot import TradingBot
from unittest.mock import AsyncMock, patch, MagicMock
import logging

async def test_handle_candle_update():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize bot
    bot = TradingBot()
    bot.current_trading_pair = "JASMY-USDT"  # Set trading pair explicitly

    # Reinitialize indicators with the trading pair
    bot.indicators.initialize_historical_data(
        "JASMY-USDT",
        pd.Series([0.0215, 0.0214, 0.0213], name='close')  # Provide initial data
    )

    # Mock the websocket manager and place_order method
    bot.websocket_manager = AsyncMock()

    # Create a mock response for successful order placement
    success_response = {
        'code': '0',
        'msg': 'success',
        'data': [{'ordId': '12345'}]
    }

    # Mock place_order to return success response
    bot.place_order = AsyncMock(return_value=success_response)

    # Create sample candle data that should trigger a trade
    sample_candle = [
        [
            1708365900000,  # timestamp
            "0.0215",       # open
            "0.0217",       # high
            "0.0214",       # low
            "0.0216",       # close
            "1000000",      # volume
            "100000",       # volCurrency
            "200000",       # volCurrencyQuote
            "1"            # confirm - Set to 1 to trigger trade
        ]
    ]

    # Initialize historical data with price above bands to trigger a long signal
    bot.candles_df = pd.DataFrame({
        'ts': [1708365800000],
        'open': [0.0214],
        'high': [0.0216],
        'low': [0.0213],
        'close': [0.0215],
        'vol': [900000],
        'volCurrency': [90000],
        'volCurrencyQuote': [180000],
        'confirm': [1],
        'trading_pair': ['JASMY-USDT']
    })

    # Set index as datetime
    bot.candles_df.index = pd.to_datetime(bot.candles_df['ts'], unit='ms')

    logger.info("Starting handle_candle_update test with confirmed candle...")

    # Test the handle_candle_update method
    await bot.handle_candle_update(sample_candle)

    # Verify that place_order was called with correct parameters
    calls = bot.place_order.call_args_list
    if calls:
        for call in calls:
            args, _ = call
            order_params = args[0]
            logger.info(f"Order parameters used: {json.dumps(order_params, indent=2)}")

            # Verify marginMode is present and correct
            assert 'marginMode' in order_params, "marginMode parameter missing in order"
            assert order_params['marginMode'] == 'isolated', "marginMode should be 'isolated'"
            assert 'tdMode' not in order_params, "tdMode should not be present in order"
    else:
        logger.warning("No orders were placed during test")

    logger.info("Handle candle update test completed.")

if __name__ == "__main__":
    asyncio.run(test_handle_candle_update())