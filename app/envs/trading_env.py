import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """Custom environment for training an RL trading agent."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Portfolio state
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.current_step = 0

        # Observation: [Close, RSI, MACD, EMA_Ratio, Sentiment, Balance, Shares]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row["Close"],
            row["rsi"],
            row["macd"],
            row["ema_ratio"],
            row["sentiment"] if not np.isnan(row["sentiment"]) else 0.0,
            self.balance,
            self.shares_held,
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]

        # Execute trade
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            self.balance -= shares_to_buy * current_price
            self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_obs = self._next_observation()
        new_portfolio_value = self.balance + self.shares_held * current_price
        reward = new_portfolio_value - self.net_worth
        self.net_worth = new_portfolio_value

        return next_obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation(), {}

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}")
