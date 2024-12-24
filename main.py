import streamlit as st
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Option Pricing Models")

st.sidebar.subheader("General Parameters")

s = float(st.sidebar.text_input("Current Asset Price", 100))
k = float(st.sidebar.text_input("Strike Price", 110))
v = float(st.sidebar.text_input("Volatility (σ)", 0.2))
r = float(st.sidebar.text_input("Risk-Free Interest Rate", 0.05))
t = float(st.sidebar.text_input("Time to Maturity (Years)", 1))

def cox_ross_rubinstein(S, K, T, R, V, N, option_type="call"):
    # Calculate parameters
    dt = T / N  # Time step
    u = np.exp(V * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(R * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset prices at maturity
    prices = np.zeros(N + 1)
    for i in range(N + 1):
        prices[i] = S * (u ** i) * (d ** (N - i))

    # Initialize option values at maturity
    option_values = np.zeros(N + 1)
    if option_type == "call":
        option_values = np.maximum(prices - K, 0)
    elif option_type == "put":
        option_values = np.maximum(K - prices, 0)

    # Work backward through the tree
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = np.exp(-R * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])

    return option_values[0]

def binomial_option_pricing(S, K, T, R, V, N, option_type="call"):
    # Calculate parameters
    dt = T / N
    u = np.exp(V * np.sqrt(dt))  # Up factor
    d = np.exp(-V * np.sqrt(dt))  # Down factor
    p = (np.exp(R * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset prices at maturity
    prices = np.zeros(N + 1)
    for i in range(N + 1):
        prices[i] = S * (u ** i) * (d ** (N - i))

    # Initialize option values at maturity
    option_values = np.zeros(N + 1)
    if option_type == "call":
        option_values = np.maximum(prices - K, 0)
    elif option_type == "put":
        option_values = np.maximum(K - prices, 0)

    # Work backward through the tree
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = np.exp(-R * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])

    return option_values[0]

def monte_carlo_option_pricing(S, K, T, R, V, n_simulations=10000, option_type="call"):
    # Simulate end-of-period prices
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal(n_simulations)  # Random normal variates
    ST = S * np.exp((R - 0.5 * V ** 2) * T + V * np.sqrt(T) * Z)  # Simulated prices

    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)  # Call payoff
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)  # Put payoff
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount payoffs back to present value
    option_price = np.exp(-R * T) * np.mean(payoffs)
    return option_price, ST

def calculate_call_option_price(S, K, V, R, T):
    d1 = (np.log(S / K) + (R + 0.5 * V ** 2) * T) / (V * np.sqrt(T))
    d2 = d1 - V * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)

def calculate_put_option_price(S, K, V, R, T):
    d1 = (np.log(S / K) + (R + 0.5 * V ** 2) * T) / (V * np.sqrt(T))
    d2 = d1 - V * np.sqrt(T)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


st.markdown("---")  # Horizontal line
st.markdown("## Binomial Option Pricing Model")
st.sidebar.subheader("Binomial Option and Cox Ross Rubenstein Parameters")
n = int(st.sidebar.slider("Number of Steps", min_value=10, max_value=500, step=10, value=100))

c_option_price = binomial_option_pricing(s, k, t, r, v, n, "call")
p_option_price = binomial_option_pricing(s, k, t, r, v, n, "put")

# Style for the Buttons
button_style = f"""
    <style>
    .center-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: auto; /* Adjust height to content */
        margin-top: 10px; /* Reduce padding between header and buttons */
    }}
    .button {{
        display: inline-block;
        padding: 15px; /* Adjust padding */
        margin: 5px; /* Reduce spacing between buttons */
        border-radius: 5px;
        font-size: 18px; /* Smaller font */
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        color: white;
        width: 100%; /* Expand to full width */
        max-width: 400px; /* Matches input width */
        box-sizing: border-box; /* Include padding in width */
    }}
    .green-button {{
        background-color: green;
    }}
    .red-button {{
        background-color: red;
    }}
    </style>
    <div class="center-container">
        <div class="button green-button">Call Price: ${c_option_price:.2f}</div>
        <div class="button red-button">Put Price: ${p_option_price:.2f}</div>
    </div>
"""

# Inject CSS and HTML into Streamlit
st.markdown(button_style, unsafe_allow_html=True)

st.markdown("---")  # Horizontal line
st.markdown("## Cox Ross Rubenstein Model")

crr_call_price = cox_ross_rubinstein(s, k, t, r, v, n, "call")
crr_put_price = cox_ross_rubinstein(s, k, t, r, v, n, "put")

# Style for the Buttons
button_style = f"""
    <style>
    .center-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: auto; /* Adjust height to content */
        margin-top: 10px; /* Reduce padding between header and buttons */
    }}
    .button {{
        display: inline-block;
        padding: 15px; /* Adjust padding */
        margin: 5px; /* Reduce spacing between buttons */
        border-radius: 5px;
        font-size: 18px; /* Smaller font */
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        color: white;
        width: 100%; /* Expand to full width */
        max-width: 400px; /* Matches input width */
        box-sizing: border-box; /* Include padding in width */
    }}
    .green-button {{
        background-color: green;
    }}
    .red-button {{
        background-color: red;
    }}
    </style>
    <div class="center-container">
        <div class="button green-button">Call Price: ${crr_call_price:.2f}</div>
        <div class="button red-button">Put Price: ${crr_put_price:.2f}</div>
    </div>
"""

# Inject CSS and HTML into Streamlit
st.markdown(button_style, unsafe_allow_html=True)

st.markdown("---")  # Horizontal line


call_price = calculate_call_option_price(s, k, v, r, t)
put_price = calculate_put_option_price(s, k, v, r, t)

st.markdown("## Black–Scholes Model")
st.subheader("Option Results")

# Style for the Buttons
button_style = f"""
    <style>
    .center-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: auto; /* Adjust height to content */
        margin-top: 10px; /* Reduce padding between header and buttons */
    }}
    .button {{
        display: inline-block;
        padding: 15px; /* Adjust padding */
        margin: 5px; /* Reduce spacing between buttons */
        border-radius: 5px;
        font-size: 18px; /* Smaller font */
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        color: white;
        width: 100%; /* Expand to full width */
        max-width: 400px; /* Matches input width */
        box-sizing: border-box; /* Include padding in width */
    }}
    .green-button {{
        background-color: green;
    }}
    .red-button {{
        background-color: red;
    }}
    </style>
    <div class="center-container">
        <div class="button green-button">Call Price: ${call_price:.2f}</div>
        <div class="button red-button">Put Price: ${put_price:.2f}</div>
    </div>
"""

# Inject CSS and HTML into Streamlit
st.markdown(button_style, unsafe_allow_html=True)

# Sidebar Inputs
st.subheader("Heatmap Parameters")
st.sidebar.subheader("Black–Scholes Heatmap Parameters")
min_spot_price = st.sidebar.number_input("Min Spot Price", value=80.0)
max_spot_price = st.sidebar.number_input("Max Spot Price", value=120.0)
min_volatility = st.sidebar.slider("Min Volatility for Heatmap (σ)", 0.01, 1.0, 0.10, 0.01)
max_volatility = st.sidebar.slider("Max Volatility for Heatmap (σ)", 0.01, 1.0, 0.30, 0.01)

# Generate Spot Price and Volatility Ranges
spot_prices = np.linspace(min_spot_price, max_spot_price, 10)
volatilities = np.linspace(min_volatility, max_volatility, 10)

# Create DataFrames for Heatmaps
call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

for i, V in enumerate(volatilities):
    for j, S in enumerate(spot_prices):
        call_prices[i, j] = calculate_call_option_price(S, k, V, r, t)
        put_prices[i, j] = calculate_put_option_price(S, k, V, r, t)

call_df = pd.DataFrame(call_prices, index=np.round(volatilities, 2), columns=np.round(spot_prices, 2))
put_df = pd.DataFrame(put_prices, index=np.round(volatilities, 2), columns=np.round(spot_prices, 2))

# Display Call Price Heatmap
st.markdown("#### Call Price Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size to make boxes bigger
sns.heatmap(
    call_df, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    cbar=True, 
    ax=ax, 
    annot_kws={"size": 8}  # Reduce font size for annotations
)
ax.set_xlabel("Spot Price")
ax.set_ylabel("Volatility (σ)")
ax.set_title("CALL")
st.pyplot(fig)

# Display Put Price Heatmap
st.markdown("#### Put Price Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size to make boxes bigger
sns.heatmap(
    put_df, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    cbar=True, 
    ax=ax, 
    annot_kws={"size": 8}  # Reduce font size for annotations
)
ax.set_xlabel("Spot Price")
ax.set_ylabel("Volatility (σ)")
ax.set_title("PUT")
st.pyplot(fig)

st.markdown("---")  # Horizontal line
st.markdown("## Monte Carlo Option Pricing Model")
st.sidebar.subheader("Monte Carlo Parameters")
n_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, step=1000, value=10000)

option_price, simulated_prices = monte_carlo_option_pricing(s, k, v, r, t, n_simulations, "call")

# Plot Simulated Prices
st.subheader("Simulated Asset Prices at Maturity for a Call")
fig, ax = plt.subplots()
ax.hist(simulated_prices, bins=50, color="lightblue", alpha=0.7)
ax.axvline(k, color="green", linestyle="--", label=f"Strike Price (K={k})")
ax.set_title("Distribution of Simulated Asset Prices at Maturity")
ax.set_xlabel("Asset Price")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

option_price, simulated_prices = monte_carlo_option_pricing(s, k, v, r, t, n_simulations, "put")

# Plot Simulated Prices
st.subheader("Simulated Asset Prices at Maturity for a Put")
fig, ax = plt.subplots()
ax.hist(simulated_prices, bins=50, color="olive", alpha=0.7)
ax.axvline(k, color="red", linestyle="--", label=f"Strike Price (K={k})")
ax.set_title("Distribution of Simulated Asset Prices at Maturity")
ax.set_xlabel("Asset Price")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)