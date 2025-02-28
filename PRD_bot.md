# Implementing a Trading Strategy Using Blofin Demo Account and Python

## Introduction

This documentation guides users through the process of using a Blofin Demo account to implement a trading strategy in a Python-based bot. The strategy involves using SMA 21 and EMA 34 bands to determine entry points. Positions are opened on the first candle after the price closes outside the bands, with configurable take-profit and stop-loss levels based on the band's position.

## Prerequisites

Before implementing the strategy, ensure the following steps are completed:

 **Configure the Trading Bot**:
   - **Timeframe**: Configurable by the user (e.g., 5m, 15m,1h, 4h, 1D).
   - **Position Size**: Configurable, with a default of 100 USD margin.
   - **Leverage**: Configurable, with a default of 3x.
   - **Position Type**: Configurable, with a default of isolated.
   - **Coin Selection**: Choose the trading pair (e.g., XRP-USDT).
   - **Risk Management**: Configure risk management parameters (e.g., TP, SL).

## Implementation

### Strategy Overview

The strategy involves the following steps:
1. Calculate SMA 21 and EMA 34 bands.
2. Monitor for price closures outside these bands.
3. Open a position on the next candle's opening after the price closes outside the band. Do not open a position later, only at the next candle's opening!!!
4. Set take-profit at the configured percentage and stop-loss just below the lower band for short positions and just above the upper band for long positions.

### Calculating SMA 21 and EMA 34 Bands

To calculate the SMA and EMA bands, fetch historical OHLC data using the Blofin API. Use websockets to receive real-time updates so that se do not miss any real price movements.


### Monitoring Price Closures Outside the Bands

Monitor real-time data for price closures outside the bands using Websocket connections for efficient data retrieval.


### Opening Positions on the Next Candle's Opening

Use scheduling libraries or server time synchronization to open positions precisely at the opening of the next candle.


### Setting Take-Profit and Stop-Loss Levels

For short positions, set the stop-loss above the band (highest wick of the last 10 candles). For long positions, set the stop-loss below the band (lowest wick of the last 10 candles). The take-profit is set at a configurable percentage from the entry point. SL will move to break-even when profit reaches a certain threshold and then trail the bands. Partial TP will be implemented at the configured amount and then a trailing TP will be implemented for the rest of position.


### Position Sizing

The position size is configurable, with a default of 100 USD margin. The contract size is adjusted to ensure the position size matches the specified margin amount.
So based on the margin size, leverage, contract size and the current price we calculate the position size and we ensure it is in the limit of the configuredmargin size.


## Conclusion

This documentation provides a comprehensive guide to implementing a trading strategy using SMA 21 and EMA 34 bands with a Blofin Demo account and Python. The strategy involves opening positions based on price closures outside the bands and setting configurable take-profit and stop-loss levels. By following this guide, users can successfully implement and test the strategy using the provided code examples and resources.

---




# Trading Bot

## Project Overview
A fully automated trading bot implementation for the Blofin exchange, running on Raspberry Pi. The bot implements a sophisticated trading strategy using SMA 21 and EMA 34 bands, with advanced dynamic stop-loss management.

## Core Components

### Main Bot Implementation (`trading_bot.py`)
- Fully automated execution
- Runs continuously on Raspberry Pi
- Handles multiple trading pairs simultaneously
- Real-time market monitoring and order execution
- Integrated with WebSocket for live data streaming
- Comprehensive position tracking and management

### Dynamic Stop-Loss System (`band_based_sl.py`)
The bot features a sophisticated band-based stop-loss management system:

1. **Initial Phase**:
   - Sets standard stop-loss at entry based on risk parameters
   - Monitors position PnL continuously

2. **Break-Even Activation**:
   - Triggers when profit reaches `BAND_SL_ACTIVATION_PROFIT` (default: $2.0)
   - Moves stop-loss to break-even with small buffer (`STOP_LOSS_BUFFER`: 0.2%)
   - Activates band-based tracking

3. **Band-Based Trailing**:
   - Uses SMA and EMA bands as dynamic support/resistance
   - For long positions:
     - Trails stop-loss using lower band
     - Adds buffer percentage for protection
   - For short positions:
     - Trails stop-loss using upper band
     - Adds buffer percentage for protection

### Trading Strategy
1. **Entry Conditions**:
   - Monitors SMA 21 and EMA 34 bands
   - Opens positions when price closes outside bands
   - Executes entry strictly on next candle opening
   - Validates position direction against current market conditions

2. **Position Management**:
   - Default margin size: 100 USD
   - Leverage: 3x (configurable)
   - Position type: Isolated by default
   - Supports multiple trading pairs simultaneously

### Technical Implementation

#### Real-Time Processing
1. **WebSocket Integration**:
   - Continuous market data streaming
   - Real-time candle updates
   - Position monitoring
   - Immediate order execution

2. **Position Tracking**:
   - Maintains detailed state for each position
   - Tracks PnL, entry prices, and stop-loss levels
   - Monitors highest unrealized PnL
   - Records break-even status

### Risk Management

#### Dynamic Risk Controls
1. **Stop-Loss Management**:
   - Initial stop-loss based on risk parameters
   - Dynamic adjustment based on profit levels
   - Band-based trailing mechanism
   - Break-even protection

2. **Position Sizing**:
   - Risk-adjusted position sizing
   - Leverage consideration
   - Market volatility adaptation
   - Account balance protection

### Monitoring and Reporting

#### Real-Time Monitoring
- Comprehensive logging system
- Position state tracking
- PnL monitoring
- Stop-loss adjustment logging

#### Email Reporting
- Daily trade summaries
- Performance metrics
- Position updates
- Risk management alerts

## Configuration System (`config.py`)
Highly configurable parameters including:
- Trading pairs selection
- Timeframe settings
- Risk parameters
- Stop-loss configurations
- Email notifications
- Position sizing rules


# Blofin Trading Bot - Complete Technical Documentation

## Project Structure
```
trading_bot/
├── config.py                 # Core configuration and environment management
├── trading_bot.py           # Main bot implementation
├── indicators.py            # Technical indicators and analysis
├── utils.py                 # Utility functions and helpers
├── websocket_manager.py     # WebSocket connection handling
├── market_scanner.py        # Market analysis and pair selection
├── risk_manager.py          # Risk management implementation
├── band_based_sl.py         # Dynamic stop-loss management
├── reporting/
│   ├── __init__.py
│   ├── trade_tracker.py     # Trade history and statistics
│   ├── email_service.py     # Email notification system
│   ├── report_generator.py  # Performance report generation
│   └── report_scheduler.py  # Automated reporting scheduler
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py
│   ├── test_trading_bot.py
│   └── test_risk_manager.py
├── .env                     # Environment variables
├── pyproject.toml          # Project dependencies and metadata
├── PRD_bot.md              # Product requirements documentation
└── README.md               # Project documentation
```

## Core Configuration (`config.py`)

### Environment Management
```python
# API Configuration
API_KEY: str
API_SECRET: str
API_PASSPHRASE: str

# Email Configuration
GMAIL_USER: str
GMAIL_PASSWORD: str
NOTIFICATION_EMAIL: str
```

### Trading Parameters
```python
# Coin Selection Modes
COIN_SELECTION_MODE: Literal["single", "multiple", "top_volume"]
SINGLE_COIN: str = "AR-USDT"
MULTIPLE_COINS: List[str]
TOP_VOLUME_COUNT: int = 10

# Timeframe Options
TIMEFRAME: Literal["1m", "3m", "5m", "15m", "30m", "1H", "4H", "1D"]

# Position Parameters
POSITION_SIZE: float = 100.0
LEVERAGE: int = 3
POSITION_TYPE: Literal["isolated", "cross"]
RISK_PER_TRADE: float = 0.01
```

### Risk Management
```python
# Stop Loss Configuration
BAND_SL_ENABLED: bool = True
BAND_SL_ACTIVATION_PROFIT: float = 2.0
BAND_SL_BUFFER: float = 0.2
TAKE_PROFIT_PERCENT: float = 4.0
SL_PERCENTAGE: float = 1.0
MAX_DRAWDOWN: float = 5.0
TRAILING_STOP: float = 0.5
```

## Core Components

### Trading Bot (`trading_bot.py`)
- Main execution loop
- Position management
- Order execution
- Real-time market monitoring
- WebSocket integration
- Error handling and recovery

### Market Scanner (`market_scanner.py`)
Features:
- Volume analysis
- Pair selection
- Market trend detection
- Volatility measurement
- Trading opportunity identification

### Risk Manager (`risk_manager.py`)
Capabilities:
- Position size calculation
- Risk exposure monitoring
- Drawdown management
- Account balance protection
- Multi-position risk correlation

### Band-Based Stop Loss (`band_based_sl.py`)
Features:
- Dynamic stop-loss adjustment
- Break-even management
- Band-based trailing
- Profit protection
- Volatility adaptation

## Technical Indicators (`indicators.py`)

### Implemented Indicators
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- Volatility calculations
- Price action analysis
- Band calculations

### Signal Generation
- Crossover detection
- Band breakout identification
- Trend strength measurement
- Entry/exit point calculation

## WebSocket Implementation (`websocket_manager.py`)

### Features
- Automatic reconnection
- Heart-beat monitoring
- Message queuing
- Error handling
- Rate limit management

### Subscriptions
- Market data
- Order updates
- Position updates
- Balance changes
- System status

## Reporting System

### Trade Tracker (`reporting/trade_tracker.py`)
Features:
- Real-time trade logging
- Performance metrics
- PnL tracking
- Trade statistics
- Historical data storage

### Email Service (`reporting/email_service.py`)
Capabilities:
- Trade notifications
- Daily summaries
- Performance reports
- Alert system
- Error notifications

### Report Generator (`reporting/report_generator.py`)
Outputs:
- Performance charts
- Trade statistics
- Risk metrics
- PnL analysis
- Position history

## Utility Functions (`utils.py`)

### Features
- API authentication
- Time synchronization
- Data formatting
- Error handling
- Logging setup

## Development Tools

### Version Control
- Git for source control
- GitHub for repository hosting

### Development Environment
- Python 3.11+
- Virtual environment management
- Development on Raspberry Pi

### Testing
- Unit tests
- Integration tests
- Performance testing
- Mock trading environment

### Deployment
- Raspberry Pi deployment
- Systemd service management
- Automatic updates
- Monitoring tools

## Dependencies

### Core Libraries
```toml
dependencies = [
    "aiohttp>=3.11.12",
    "numpy>=2.2.3",
    "pandas>=2.2.3",
    "python-dotenv>=1.0.1",
    "websockets>=15.0",
    "matplotlib>=3.8.0",
    "cryptography>=42.0.0",
    "aioschedule>=0.5.2",
    "pytz>=2024.1"
]
```

## API Integration

### Blofin API
- REST API endpoints
- WebSocket connections
- Rate limit management
- Authentication handling
- Error handling

## Monitoring and Logging

### Logging System
- File-based logging
- Console output
- Error tracking
- Performance monitoring
- System status

### Monitoring Tools
- Resource usage
- Network connectivity
- API status
- System health
- Performance metrics

## Future Enhancements

### Planned Features
1. Machine Learning Integration
   - Pattern recognition
   - Trend prediction
   - Risk assessment
   - Automated parameter optimization

2. Advanced Analytics
   - Market correlation analysis
   - Risk factor analysis
   - Performance attribution
   - Portfolio optimization

3. User Interface
   - Web dashboard
   - Mobile app
   - Real-time monitoring
   - Configuration interface

4. Additional Functionality
   - Multi-exchange support
   - Additional trading strategies
   - Enhanced risk management
   - Advanced reporting features


