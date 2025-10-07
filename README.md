# Stock Trading Reinforcement Learning Agent

## 1. Overview
This project develops a reinforcement learning agent for algorithmic stock trading. It combines historical stock price data, technical indicators, and financial news sentiment to train an AI model that autonomously learns buy/hold/sell strategies.

The project demonstrates a full machine learning workflow — from data acquisition and feature engineering to custom environment design, agent training, and deployment — simulating real-world quantitative finance applications.

The goal is to maximize portfolio returns using Proximal Policy Optimization (PPO), achieving intelligent trading decisions based on both market signals and sentiment insights.

---

## 2. Project Structure
stock-rl-trader/
│
├── data/
│   ├── raw/                      # Original Kaggle datasets
│   └── processed/                # Cleaned and enriched dataset
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_agent_training.ipynb
│   ├── 03_agent_evaluation.ipynb
│   └── test.ipynb
│
├── models/
│   └── ppo_trading_agent.zip     # Trained PPO model (local only)
│
├── app/
│   ├── envs/
│   │   └── trading_env.py        # Custom Gymnasium trading environment
│   └── dashboard/
│       └── app.py                # Streamlit dashboard for visualization
│
├── utils/
│   ├── indicators.py             # RSI, MACD, EMA calculations
│   └── sentiment.py              # News sentiment scoring
│
├── requirements.txt
├── README.md
└── LICENSE

---

## 3. Data Sources
1. Multiple Stock Prices by Industry  
   https://www.kaggle.com/datasets/chayanonc/multiple-stock-prices-by-industry-updated-daily

2. Daily Financial News for 6000+ Stocks  
   https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

---

## 4. Technical Workflow

### Data Preprocessing
- Cleaned and merged stock prices with sentiment data.
- Converted wide-format stock data to long format.
- Handled missing values and aligned time periods between datasets.

### Feature Engineering
- Computed technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Exponential Moving Averages (EMA10, EMA30)
  - EMA Ratio (EMA10 / EMA30)
- Extracted sentiment scores from financial headlines using TextBlob.
- Saved enriched dataset to data/processed/final_dataset.csv.

### Environment Design
Developed a custom Gymnasium environment (StockTradingEnv) simulating a stock trading process:
- State vector: [Close, RSI, MACD, EMA_ratio, Sentiment, Balance, Shares]
- Actions: 0 = Hold, 1 = Buy, 2 = Sell
- Reward: Change in total portfolio value per timestep
- Goal: Maximize cumulative profit while managing risk.

### Model Training
Trained an RL agent using Proximal Policy Optimization (PPO) with Stable-Baselines3.  
Hyperparameters optimized for stable learning and exploration.

Parameter | Value
-----------|--------
Learning Rate | 3e-4
Discount Factor (γ) | 0.99
Entropy Coefficient | 0.01
n_steps | 128
Policy | MLP (2-layer neural network)

Training command:
python notebooks/02_agent_training.ipynb

Model saved to:
models/ppo_trading_agent.zip

### Model Evaluation
The trained agent was evaluated on unseen data.
- Plotted portfolio value vs. time (equity curve)
- Analyzed buy/sell/hold actions
- Computed key metrics:

Metric | Description | Example
--------|--------------|---------
Total Return | % increase in portfolio value | 18.4%
Sharpe Ratio | Risk-adjusted return | 1.34
Cumulative Reward | Learning stability | Upward trend

### Visualization and Dashboard
Built an interactive Streamlit dashboard for real-time simulation and visualization of the agent's trading performance.

Run locally from project root:
streamlit run app/dashboard/app.py

Features:
- Select stock ticker
- Adjust starting balance
- Run trained model
- Visualize portfolio performance, buy/sell actions, and metrics (Sharpe, returns)

---

## 5. Model Performance

Example PPO Agent Results

Metric | Value
--------|--------
Total Return | 18.42%
Sharpe Ratio | 1.34
Cumulative Reward | +15,300
Average Profit per Trade | +0.67%

Key Insights:
- The agent learned profitable patterns from technical and sentiment data.
- EMA ratios helped identify short-term trend reversals.
- Sentiment analysis improved reaction to market news shocks.

---

## 6. Deployment

### Streamlit Dashboard
Interactive visualization app for exploring the trained model.

Run locally:
streamlit run app/dashboard/app.py

Example Interface:
- Equity Curve: Portfolio value growth over time
- Buy/Sell Actions: Green = Buy, Red = Sell
- Performance Metrics: Sharpe ratio, total return, and balance history

---

## 7. Environment Setup

Create and activate virtual environment:
python -m venv .venv  
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Main requirements:
pandas  
numpy  
textblob  
matplotlib  
plotly  
gymnasium==0.29.1  
stable-baselines3[extra]  
streamlit

---

## 8. Future Enhancements
- Integrate transaction fees and slippage for realism.
- Expand to multi-stock portfolio management.
- Compare PPO vs. DQN vs. A2C performance.
- Deploy Streamlit app on Hugging Face or Streamlit Cloud.
- Add risk management constraints and drawdown tracking.
- Incorporate macroeconomic sentiment data.

---

## 9. License
This project is licensed under the MIT License.  
You are free to use, modify, and distribute it with proper attribution.

---

## 10. Author
**Oanh Nguyen**  
[LinkedIn](https://www.linkedin.com/in/oanh-nguyen-1021/)  
[GitHub](https://github.com/oanhtkn)