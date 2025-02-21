import logging
import aiohttp
from typing import List, Dict, Optional
from decimal import Decimal

class MarketScanner:
    def __init__(self, config):
        """Initialize MarketScanner with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def get_all_tickers(self) -> Optional[List[Dict]]:
        """Fetch all trading pairs tickers from Blofin"""
        try:
            url = f"{self.config.REST_URL}/api/v1/market/tickers"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()

                    if data['code'] == '0' and data['data']:
                        return data['data']
                    else:
                        self.logger.error(f"Failed to fetch tickers: {data.get('msg', 'Unknown error')}")
                        return None

        except Exception as e:
            self.logger.error(f"Error fetching tickers: {str(e)}")
            return None

    async def get_trading_pairs(self) -> List[str]:
        """Get trading pairs based on configured selection mode"""
        try:
            if self.config.COIN_SELECTION_MODE == "single":
                return [self.config.SINGLE_COIN]

            elif self.config.COIN_SELECTION_MODE == "multiple":
                if not self.config.MULTIPLE_COINS:
                    self.logger.warning("No multiple coins configured, defaulting to single coin mode")
                    return [self.config.SINGLE_COIN]
                return self.config.MULTIPLE_COINS

            elif self.config.COIN_SELECTION_MODE == "top_volume":
                return await self.get_top_volume_pairs()

            else:
                self.logger.error(f"Invalid selection mode: {self.config.COIN_SELECTION_MODE}")
                return [self.config.SINGLE_COIN]

        except Exception as e:
            self.logger.error(f"Error getting trading pairs: {str(e)}")
            return [self.config.SINGLE_COIN]

    async def get_top_volume_pairs(self) -> List[str]:
        """Get top volume trading pairs"""
        try:
            tickers = await self.get_all_tickers()
            if not tickers:
                self.logger.error("Failed to fetch tickers for top volume analysis")
                return [self.config.SINGLE_COIN]

            # Sort tickers by 24h volume
            sorted_tickers = sorted(
                tickers,
                key=lambda x: float(x.get('vol24h', 0)),
                reverse=True
            )

            # Extract top N pairs with volume info
            top_pairs = []
            for ticker in sorted_tickers[:self.config.TOP_VOLUME_COUNT]:
                if 'instId' in ticker:
                    top_pairs.append(ticker['instId'])

            if not top_pairs:
                self.logger.warning("No valid pairs found in top volume analysis")
                return [self.config.SINGLE_COIN]

            # Enhanced volume information logging
            self.logger.info("\n📊 Top Trading Pairs by 24h Volume:")
            self.logger.info("=" * 80)
            self.logger.info(f"Selection Mode: {self.config.COIN_SELECTION_MODE}")
            self.logger.info(f"Number of Pairs: {self.config.TOP_VOLUME_COUNT}")
            self.logger.info("=" * 80)

            for i, ticker in enumerate(sorted_tickers[:self.config.TOP_VOLUME_COUNT], 1):
                volume_24h = float(ticker.get('vol24h', 0))
                current_vol = float(ticker.get('vol', 0))  # Current timeframe volume
                price = float(ticker.get('last', 0))
                price_change = float(ticker.get('priceChangePercent', 0))

                self.logger.info(
                    f"{i}. {ticker['instId']}\n"
                    f"   💰 24h Volume: ${volume_24h:,.2f}\n"
                    f"   📈 Current Price: ${price:,.6f} ({price_change:+.2f}%)\n"
                    f"   📊 Current Period Volume: ${current_vol:,.2f}\n"
                    f"   {'=' * 40}"
                )

            return top_pairs

        except Exception as e:
            self.logger.error(f"Error getting top volume pairs: {str(e)}")
            return [self.config.SINGLE_COIN]

    async def validate_trading_pairs(self, pairs: List[str]) -> List[str]:
        """Validate trading pairs against available instruments"""
        try:
            url = f"{self.config.REST_URL}/api/v1/market/instruments"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()

                    if data['code'] == '0' and data['data']:
                        valid_instruments = {inst['instId'] for inst in data['data']}
                        validated_pairs = [
                            pair for pair in pairs 
                            if pair in valid_instruments
                        ]

                        if not validated_pairs:
                            self.logger.warning("No valid trading pairs found after validation")
                            return [self.config.SINGLE_COIN]

                        invalid_pairs = set(pairs) - set(validated_pairs)
                        if invalid_pairs:
                            self.logger.warning(f"Invalid trading pairs removed: {invalid_pairs}")

                        return validated_pairs
                    else:
                        self.logger.error(f"Failed to fetch instruments: {data.get('msg', 'Unknown error')}")
                        return [self.config.SINGLE_COIN]

        except Exception as e:
            self.logger.error(f"Error validating trading pairs: {str(e)}")
            return [self.config.SINGLE_COIN]