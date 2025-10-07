import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.envs.trading_env import StockTradingEnv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO
import gymnasium as gym
import os


st.set_page_config(page_title="AI Stock Trading Agent", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_dataset.csv")
    return df

df = load_data()
tickers = sorted(df["Ticker"].unique())

#Side bar
st.sidebar.header("Simulation Controls")
selected_ticker = st.sidebar.selectbox("Select Stock", tickers)
model_path = "models/ppo_trading_agent.zip"
start_balance = st.sidebar.number_input("Initial Balance ($)", 1000, 50000, 10000)
run_button = st.sidebar.button("Run Simulation")

#Title
st.title("🤖 Reinforcement Learning Stock Trader Dashboard")
st.write(f"**Trained PPO Agent | Ticker:** {selected_ticker}")

#Run Simulation when clicked
if run_button and os.path.exists(model_path):
    df_ticker = df[df["Ticker"] == selected_ticker].reset_index(drop=True)
    env = StockTradingEnv(df_ticker, initial_balance=start_balance)
    #env = gym.wrappers.FlattenObservation(env)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    net_worths, prices, actions, rewards = [], [], [], []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        net_worths.append(env.net_worth)
        prices.append(env.df.iloc[env.current_step]["Close"])
        actions.append(action)
        rewards.append(reward)

    #Metrics
    returns = pd.Series(net_worths).pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    total_return = (net_worths[-1] / net_worths[0]) - 1

    st.metric("📈 Final Portfolio Value", f"${net_worths[-1]:,.2f}")
    st.metric("💰 Total Return", f"{total_return*100:.2f}%")
    st.metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}")

    # Plot Equity Curve
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=net_worths, mode='lines', name='Agent Net Worth'))
    fig1.update_layout(title="Portfolio Value Over Time", xaxis_title="Step", yaxis_title="Value ($)")
    st.plotly_chart(fig1, use_container_width=True)

    #Plot Buy/Sell Points
    buy_points = [i for i, a in enumerate(actions) if a == 1]
    sell_points = [i for i, a in enumerate(actions) if a == 2]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=prices, mode='lines', name='Stock Price'))
    fig2.add_trace(go.Scatter(
        x=buy_points, y=[prices[i] for i in buy_points],
        mode='markers', marker=dict(color='green', size=8), name='Buy'
    ))
    fig2.add_trace(go.Scatter(
        x=sell_points, y=[prices[i] for i in sell_points],
        mode='markers', marker=dict(color='red', size=8), name='Sell'
    ))
    fig2.update_layout(title="Agent Buy/Sell Decisions", xaxis_title="Step", yaxis_title="Price ($)")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Click **Run Simulation** to test your trained agent.")
