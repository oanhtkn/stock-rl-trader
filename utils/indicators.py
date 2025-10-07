import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["ema_10"] = ta.trend.EMAIndicator(df["Close"], window=10).ema_indicator()
    df["ema_30"] = ta.trend.EMAIndicator(df["Close"], window=30).ema_indicator()
    df["ema_ratio"] = df["ema_10"] / df["ema_30"]
    return df.fillna(0)
