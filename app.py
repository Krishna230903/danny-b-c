import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import calendar

# --- Helper Functions with Caching ---

@st.cache_data(ttl=1800)  # Cache data for 30 minutes
def get_stock_data(ticker_symbol):
    """
    Fetches stock data and calculates volatility.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if 'longName' not in info or info['longName'] is None:
            # Return None for all values if the ticker is invalid
            return None, None, None

        hist = ticker.history(period="1y")
        if hist.empty:
            return None, None, None
            
        S0 = hist['Close'].iloc[-1]
        long_name = info['longName']

        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        sigma = log_returns.std() * np.sqrt(252)
        
        return S0, sigma, long_name
    except Exception:
        # Return None for all values if anything fails
        return None, None, None

def get_indian_risk_free_rate():
    """
    Returns a fixed rate as a proxy for the Indian risk-free rate.
    """
    return 0.07

# --- CORE CALCULATION FUNCTIONS ---

def calculate_option_price_custom(S0, K, T, r, sigma, option_type):
    """Calculates the option price based on the user's custom single-step binomial model."""
    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    u = 1 + sigma
    d = 1 / u
    
    if u == d:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    price_up = S0 * u
    price_down = S0 * d

    prob_up = ((1 + r * T) - d) / (u - d)
    prob_down = 1 - prob_up

    if option_type == 'Call':
        payoff_up = max(0, price_up - K)
        payoff_down = max(0, price_down - K)
    else:  # Put
        payoff_up = max(0, K - price_up)
        payoff_down = max(0, K - price_down)

    total_value_final_node = (payoff_up * prob_up) + (payoff_down * prob_down)
    option_price = total_value_final_node / (1 + r * T)
    
    return option_price, u, d, prob_up, prob_down, price_up, price_down, payoff_up, payoff_down

def calculate_greeks(S0, K, T, r, sigma, option_type):
    """
    Calculates option Greeks using the finite difference method.
    """
    dS = S0 * 0.01
    dSigma = 0.01
    dT = 1 / 365.0
    dR = 0.01

    base_price, *_ = calculate_option_price_custom(S0, K, T, r, sigma, option_type)
    price_plus_S, *_ = calculate_option_price_custom(S0 + dS, K, T, r, sigma, option_type)
    price_minus_S, *_ = calculate_option_price_custom(S0 - dS, K, T, r, sigma, option_type)
    price_plus_sigma, *_ = calculate_option_price_custom(S0, K, T, r, sigma + dSigma, option_type)
    price_minus_T, *_ = calculate_option_price_custom(S0, K, T - dT, r, sigma, option_type)
    price_plus_r, *_ = calculate_option_price_custom(S0, K, T, r + dR, sigma, option_type)

    delta = (price_plus_S - price_minus_S) / (2 * dS)
    gamma = (price_plus_S - 2 * base_price + price_minus_S) / (dS ** 2)
    vega = (price_plus_sigma - base_price) / (dSigma * 100)
    theta = (price_minus_T - base_price) / dT
    rho = (price_plus_r - base_price) / (dR * 100)
    
    return delta, gamma, vega, theta, rho

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(layout="wide")
st.title("Indian Market Binomial Options Calculator")
st.markdown("Enter an NSE stock ticker, and the app will fetch live data to price the option using your custom formulas.")

st.subheader("1. Select a Stock")
ticker_input = st.text_input(
    "Enter Stock Ticker (e.g., RELIANCE, INFY, TCS)", 
    "RELIANCE"
).upper()

if ticker_input:
    if not (ticker_input.endswith('.NS') or ticker_input.endswith('.BO')):
        ticker_symbol = ticker_input + '.NS'
        st.info(f"Appended '.NS' for the Indian market. Searching for: {ticker_symbol}")
    else:
        ticker_symbol = ticker_input

    S0, sigma, long_name = get_stock_data(ticker_symbol)

    if S0 is None:
        st.error(f"Invalid or unsupported ticker symbol: {ticker_symbol}. Please check the symbol and try again.")
    else:
        st.header(f"Selected: {long_name} ({ticker_symbol})")
        r = get_indian_risk_free_rate()

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Stock Price (S₀)", f"₹{S0:,.2f}")
        col2.metric("Annualized Volatility (σ)", f"{sigma:.2%}")
        col3.metric("Risk-Free Rate (r)", f"{r:.2%}")
        st.caption("Note: Risk-free rate is a fixed 7.0% proxy for the Indian 10-year government bond yield.")

        st.divider()

        st.subheader("2. Select Option Parameters")
        
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            option_type = st.selectbox("Option Type", ('Call', 'Put'))

        with sub_col2:
            # --- CHANGE: Input for number of days instead of selecting a date ---
            days_to_expiry = st.number_input(
                "Days to Expiration", 
                min_value=1, 
                max_value=730,  # Max of ~2 years
                value=30, 
                step=1,
                help="Enter the number of days until the option expires."
            )

        # Calculate Time to Expiration (T) directly from the number of days
        T = days_to_expiry / 365.0

        with sub_col3:
            # --- CHANGE: Manual number input for Strike Price ---
            # Set a sensible default value close to the current stock price
            default_strike = round(S0 / 5) * 5 # Round to nearest 5
            K = st.number_input(
                "Strike Price (K)", 
                min_value=0.0,
                value=float(default_strike), 
                step=1.0, 
                format="%.2f"
            )
        
        st.info(f"Time to Expiration (T) = **{T:.3f} years** ({days_to_expiry} days)", icon="⏳")
        st.divider()
        
        st.header("3. Calculation Results")
        (option_price, u, d, prob_up, prob_down, 
         price_up, price_down, payoff_up, payoff_down) = calculate_option_price_custom(S0, K, T, r, sigma, option_type)

        st.metric(label=f"Calculated {option_type} Option Price", value=f"₹{option_price:,.4f}")

        delta, gamma, vega, theta, rho = calculate_greeks(S0, K, T, r, sigma, option_type)
        st.subheader("Option Greeks (Estimates)")
        
        greek_col1, greek_col2, greek_col3, greek_col4, greek_col5 = st.columns(5)
        greek_col1.metric("Delta", f"{delta:.4f}")
        greek_col2.metric("Gamma", f"{gamma:.4f}")
        greek_col3.metric("Vega", f"{vega:.4f}")
        greek_col4.metric("Theta", f"{theta:.4f}")
        greek_col5.metric("Rho", f"{rho:.4f}")

        st.divider()

        st.subheader("Intermediate Values (Your Formulas)")
        if not (0 <= prob_up <= 1):
            st.warning(f"Arbitrage Opportunity Detected! The calculated probability ({prob_up:.2f}) is outside the valid [0, 1] range. Results may be unreliable.")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write(f"**Up Factor (u):** `{u:.4f}`")
            st.write(f"**Down Factor (d):** `{d:.4f}`")
            st.write(f"**Stock Price (Up):** `₹{price_up:,.2f}`")
            st.write(f"**Stock Price (Down):** `₹{price_down:,.2f}`")
        with res_col2:
            st.write(f"**Probability of Up Move (p):** `{prob_up:.4f}`")
            st.write(f"**Probability of Down Move (1-p):** `{prob_down:.4f}`")
            st.write(f"**Payoff (Up):** `₹{payoff_up:,.2f}`")
            st.write(f"**Payoff (Down):** `₹{payoff_down:,.2f}`")

