import asyncio
from trading_bot import TradingBot

async def test_report_generation():
    print("Initializing bot for report testing...")
    bot = TradingBot()
    await bot.initialize()  # This will set up the necessary components
    
    print("Generating and sending test report...")
    await bot.send_test_report()
    
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_report_generation())
