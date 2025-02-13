-- Create stock price data table
CREATE TABLE IF NOT EXISTS bx_stock_prices (
    Date TEXT PRIMARY KEY,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL,
    Adj_Close REAL,
    Volume INTEGER,
    Daily_Return REAL,
    Log_Return REAL,
    SMA_50 REAL,
    SMA_200 REAL,
    EMA_50 REAL,
    Volatility_30 REAL,
    RSI_14 REAL
);

-- Create financial statement data table
CREATE TABLE IF NOT EXISTS bx_financials (
    Date TEXT PRIMARY KEY,
    Revenue REAL,
    Net_Income REAL,
    AUM REAL,
    Cash_Flow REAL,
    Debt REAL
);

-- Create macroeconomic indicators table
CREATE TABLE IF NOT EXISTS bx_macroeconomic (
    Date TEXT PRIMARY KEY,
    Interest_Rate REAL,
    Inflation REAL,
    Credit_Spread REAL,
    Market_Volatility REAL
);

-- Create liquidity forecast table
CREATE TABLE IF NOT EXISTS bx_liquidity_forecast (
    Date TEXT PRIMARY KEY,
    Forecasted_Capital_Calls REAL,
    Forecasted_Distributions REAL,
    Liquidity_Buffer REAL,
    Stress_Test_Scenario TEXT
);
