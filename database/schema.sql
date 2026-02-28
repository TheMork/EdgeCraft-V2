-- Table for individual trades (Level 1 Data)
CREATE TABLE trades (
    symbol SYMBOL,
    side SYMBOL,
    price DOUBLE,
    amount DOUBLE,
    id STRING,
    timestamp TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY MONTH;

-- Table for funding rates
CREATE TABLE funding_rates (
    symbol SYMBOL,
    rate DOUBLE,
    timestamp TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY MONTH;
