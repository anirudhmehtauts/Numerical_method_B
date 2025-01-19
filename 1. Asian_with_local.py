import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import os
from tqdm import tqdm
import math
pd.set_option('display.max_columns', None)


def get_option_price(df_date, T, K, implied_vol, B, D):
    """Helper function to get option price at given T and K using Black-Scholes"""
    try:
        S = df_date['UnderlyingPrice'].iloc[0]
        if all(v is not None for v in [implied_vol, B, D]):
            return black_scholes(S, K, implied_vol, B, D, T, 'call')
        return None
    except Exception as e:
        print(f"Option price calculation failed: {e}")
        return None


def calculate_local_volatility(time, spot_price, K, option_prices, r, q, eps=1e-6):
    """
    Calculate local volatility using the Dupire formula
    """
    try:
        dT = time * 0.01
        dK = K * 0.01

        C = option_prices['base']
        C_T_up = option_prices['T_up']
        C_K_up = option_prices['K_up']
        C_K_down = option_prices['K_down']
        C_K2_up = option_prices['K2_up']

        dC_dT = (C_T_up - C) / dT
        dC_dK = (C_K_up - C_K_down) / (2 * dK)
        d2C_dK2 = (C_K2_up - 2 * C_K_up + C) / (dK * dK)

        numerator = 2 * (dC_dT + q * C + (r - q) * K * dC_dK)
        denominator = K * K * d2C_dK2

        if abs(denominator) < eps:
            denominator = eps if denominator >= 0 else -eps

        local_var = numerator / denominator
        local_var = max(local_var, 0)

        return np.sqrt(local_var)

    except Exception as e:
        print(f"Local volatility calculation failed: {e}")
        return None


def original_interpolation_logic(df_date, T, K, param):
    """Original interpolation logic for surface"""
    if df_date.empty:
        return None

    df_clean = df_date[df_date[param].notna()]
    if df_clean.empty:
        return None

    unique_T = sorted(df_clean['T'].unique())
    if len(unique_T) < 2:
        return None

    T_idx = np.searchsorted(unique_T, T)
    T_idx = min(max(1, T_idx), len(unique_T) - 1)
    T_lower, T_upper = unique_T[T_idx - 1], unique_T[T_idx]

    lower_slice = (df_clean[df_clean['T'] == T_lower]
                   .drop_duplicates('StrikePrice')
                   .sort_values('StrikePrice'))
    upper_slice = (df_clean[df_clean['T'] == T_upper]
                   .drop_duplicates('StrikePrice')
                   .sort_values('StrikePrice'))

    if len(lower_slice) < 4 or len(upper_slice) < 4:
        return None

    try:
        cs_lower = CubicSpline(lower_slice['StrikePrice'], lower_slice[param])
        cs_upper = CubicSpline(upper_slice['StrikePrice'], upper_slice[param])

        val_lower = cs_lower(K)
        val_upper = cs_upper(K)

        result = val_lower + (val_upper - val_lower) * (T - T_lower) / (T_upper - T_lower)
        return result if np.isfinite(result) else None

    except Exception as e:
        print(f"Interpolation error for {param}: {e}")
        return None


def interpolate_surface(df_date, T, K, param):
    if df_date.empty:
        return None

    if param != 'ImpliedVol':
        return original_interpolation_logic(df_date, T, K, param)

    try:
        implied_vol = original_interpolation_logic(df_date, T, K, param)
        if implied_vol is None:
            return None

        dT = T * 0.01
        dK = K * 0.01

        B = original_interpolation_logic(df_date, T, K, 'B')
        D = original_interpolation_logic(df_date, T, K, 'D')

        if B is None or D is None:
            return implied_vol

        option_prices = {}

        # Calculate option prices at different points
        option_prices['base'] = get_option_price(df_date, T, K, implied_vol, B, D)
        option_prices['T_up'] = get_option_price(df_date, T + dT, K, implied_vol, B, D)
        option_prices['K_up'] = get_option_price(df_date, T, K + dK, implied_vol, B, D)
        option_prices['K_down'] = get_option_price(df_date, T, K - dK, implied_vol, B, D)
        option_prices['K2_up'] = get_option_price(df_date, T, K + 2 * dK, implied_vol, B, D)

        if any(price is None for price in option_prices.values()):
            return implied_vol

        spot_price = df_date['UnderlyingPrice'].iloc[0]

        if T == 0:
            r = q = 0

        else:
            r = -np.log(B) / T
            q = -np.log(D) / T

        local_vol = calculate_local_volatility(T, spot_price, K, option_prices, r, q)
        #print(local_vol, implied_vol)
        #return local_vol if local_vol is not None or 0 or (isinstance(local_vol, float) and math.isnan(local_vol)) else implied_vol
        vol_condn = local_vol > 0 and local_vol < 0.50

        if vol_condn:
            vol = local_vol
        else:
            vol = implied_vol
        return vol
    except Exception as e:
        print(f"Local volatility calculation failed: {e}")
        return implied_vol


def black_scholes(S, K, sigma, B, D, T, option_type):
    d1 = (np.log(S * D / (K * B)) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * D * norm.cdf(d1) - K * B * norm.cdf(d2)
    else:
        return K * B * norm.cdf(-d2) - S * D * norm.cdf(-d1)


def simulate_paths_with_surface(df, S0, K, T, n_paths, n_steps, initial_date, vol_override=None):
    np.random.seed(0)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    dates = sorted(df['DataDate'].unique())
    current_date_idx = dates.index(initial_date)

    times = T - (np.arange(1, n_steps + 1) * dt)
    Z = np.random.standard_normal((n_paths, n_steps))

    for t in range(1, n_steps + 1):
        date_data = df[df['DataDate'] == dates[current_date_idx]]

        if vol_override is not None:
            sigma = vol_override
        else:
            sigma = interpolate_surface(date_data, times[t - 1], K, 'ImpliedVol')

        B = interpolate_surface(date_data, times[t - 1], K, 'B')
        D = interpolate_surface(date_data, times[t - 1], K, 'D')

        if sigma is None or (isinstance(sigma, float) and math.isnan(sigma)) or sigma < 0 or sigma > 0.60:
            sigma = last_valid_sigma if 'last_valid_sigma' in locals() else 0.2

        last_valid_sigma = sigma

        if any(v is None for v in [sigma, B, D]):
            print(sigma, B, D)
            print(date_data)
            raise ValueError(f"Surface interpolation failed at step {t}")

        paths[:, t] = paths[:, t - 1] * np.exp(
            (np.log(D / B) - 0.5 * sigma ** 2) * dt +
            sigma * np.sqrt(dt) * Z[:, t - 1]
        )

        if current_date_idx + 1 < len(dates):
            current_date_idx += 1

    return paths

def asian_option_price_and_greeks_surface(df, S, K, T, option_type, initial_date, n_paths=10000, n_steps=252,
                                          bump=0.01):
    final_date_data = df[df['DataDate'] == df['DataDate'].max()]
    final_B = interpolate_surface(final_date_data, T, K, 'B')

    if final_B is None:
        raise ValueError("Failed to interpolate final discount factor")

    def asian_payoff(paths, strike):
        averages = np.mean(paths, axis=1)
        if option_type.lower() == 'call':
            return np.maximum(averages - strike, 0)
        return np.maximum(strike - averages, 0)

    base_paths = simulate_paths_with_surface(df, S, K, T, n_paths, n_steps, initial_date)
    base_payoffs = asian_payoff(base_paths, K)
    base_price = final_B * np.mean(base_payoffs)

    spot_up = S * (1 + bump)
    spot_down = S * (1 - bump)

    up_paths = simulate_paths_with_surface(df, spot_up, K, T, n_paths, n_steps, initial_date)
    down_paths = simulate_paths_with_surface(df, spot_down, K, T, n_paths, n_steps, initial_date)

    up_payoffs = asian_payoff(up_paths, K)
    down_payoffs = asian_payoff(down_paths, K)

    up_price = final_B * np.mean(up_payoffs)
    down_price = final_B * np.mean(down_payoffs)

    delta = (up_price - down_price) / (spot_up - spot_down)
    gamma = (up_price - 2 * base_price + down_price) / ((bump * S) ** 2)

    vol = interpolate_surface(df[df['DataDate'] == initial_date], T, K, 'ImpliedVol')
    if vol is None:
        raise ValueError("Failed to interpolate volatility")

    vol_up = vol * (1 + bump)
    vol_down = vol * (1 - bump)

    paths_vol_up = simulate_paths_with_surface(df, S, K, T, n_paths, n_steps, initial_date, vol_override=vol_up)
    paths_vol_down = simulate_paths_with_surface(df, S, K, T, n_paths, n_steps, initial_date, vol_override=vol_down)

    payoffs_vol_up = asian_payoff(paths_vol_up, K)
    payoffs_vol_down = asian_payoff(paths_vol_down, K)

    price_vol_up = final_B * np.mean(payoffs_vol_up)
    price_vol_down = final_B * np.mean(payoffs_vol_down)

    vega = (price_vol_up - price_vol_down) / ((vol_up - vol_down) * 100)

    return base_price, delta, gamma, vega

def calculate_european_greeks(S, K, sigma, B, D, T):
    if T <= 0:
        return 0, 0, 0

    d1 = (np.log(S * D / (K * B)) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = D * norm.cdf(d1)
    gamma = (D * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * D * np.sqrt(T) * norm.pdf(d1)

    return delta, gamma, vega

def calculate_pnl(df, option_params):
    if df.empty or option_params.empty:
        print("Error: Empty dataframe provided")
        return pd.DataFrame()

    required_columns = ['DataDate', 'T', 'StrikePrice', 'UnderlyingPrice']
    if not all(col in df.columns for col in required_columns):
        print("Error: Missing required columns in dataframe")
        return pd.DataFrame()

    if not isinstance(df['DataDate'].iloc[0], pd.Timestamp):
        df['DataDate'] = pd.to_datetime(df['DataDate'], format='mixed')
    df['DataDate'] = df['DataDate'].dt.strftime('%Y-%m-%d')

    df = df.sort_values(['DataDate', 'T', 'StrikePrice'])

    ref_date = df['DataDate'].max()
    ref_data = df[df['DataDate'] == ref_date]
    dates = sorted(df[df['DataDate'] < ref_date]['DataDate'].unique(), reverse=True)

    all_results = []

    option_maturity = option_params['Maturity(years)'].iloc[0]
    option_type = option_params['CallPut'].iloc[0].lower()
    strike = option_params['Strike'].iloc[0]
    option_style = option_params['OptionType'].iloc[0].lower()

    current_price = ref_data['UnderlyingPrice'].iloc[0]
    B_current = interpolate_surface(ref_data, option_maturity, strike, 'B')
    D_current = interpolate_surface(ref_data, option_maturity, strike, 'D')
    sigma_current = interpolate_surface(ref_data, option_maturity, strike, 'ImpliedVol')

    if any(v is None for v in [B_current, D_current, sigma_current]):
        print(f"Skipping {option_style} option due to missing initial data")
        return pd.DataFrame()

    if option_style == 'european':
        current_value = black_scholes(current_price, strike, sigma_current, B_current, D_current,
                                      option_maturity, option_type)
        prev_delta, prev_gamma, prev_vega = calculate_european_greeks(
            current_price, strike, sigma_current, B_current, D_current, option_maturity)
    else:
        current_value, prev_delta, prev_gamma, prev_vega = asian_option_price_and_greeks_surface(
            df, current_price, strike, option_maturity, option_type, ref_date)

    prev_price = current_price
    prev_sigma = sigma_current
    prev_value = current_value

    for date in dates:
        if option_style == 'european':
            break
        next_data = df[df['DataDate'] == date]
        next_price = next_data['UnderlyingPrice'].iloc[0]

        T_prev = option_maturity
        B_prev = interpolate_surface(next_data, T_prev, strike, 'B')
        D_prev = interpolate_surface(next_data, T_prev, strike, 'D')
        sigma_prev = interpolate_surface(next_data, T_prev, strike, 'ImpliedVol')

        if all(v < 0.0001 for v in [B_prev, D_prev, sigma_prev]):
            continue

        if sigma_prev is None:
            sigma_prev = last_valid_sigma if 'last_valid_sigma' in locals() else 0.2

        last_valid_sigma = sigma_prev

        if any(v is None for v in [B_prev, D_prev, sigma_prev]):
            continue

        if option_style == 'european':
            next_value = black_scholes(next_price, strike, sigma_prev, B_prev, D_prev,
                                       T_prev, option_type)
            delta, gamma, vega = calculate_european_greeks(next_price, strike, sigma_prev,
                                                           B_prev, D_prev, T_prev)
        else:
            next_value, delta, gamma, vega = asian_option_price_and_greeks_surface(
                df, next_price, strike, T_prev, option_type, date)

        full_reval_pnl = next_value - prev_value
        daily_log_return = np.log(next_price / prev_price)
        daily_vol_log_return = np.log(sigma_prev / prev_sigma)

        delta_pnl = daily_log_return * prev_price * prev_delta
        gamma_pnl = (0.5 * prev_gamma * (prev_price * daily_log_return) ** 2 +
                     0.5 * prev_delta * prev_price * (daily_log_return ** 2))
        vega_pnl = daily_vol_log_return * prev_vega * prev_sigma

        theoretical_pnl = delta_pnl + gamma_pnl + vega_pnl
        unexplained_pnl = -(theoretical_pnl - full_reval_pnl)

        print(date, sigma_prev, vega_pnl)

        all_results.append({
            'DataDate': date,
            'OptionType': option_style,
            'Strike': strike,
            'FullRevalPnL': full_reval_pnl,
            'TheoreticalPnL': theoretical_pnl,
            'UnexplainedPnL': unexplained_pnl,
            'DeltaPnL': delta_pnl,
            'GammaPnL': gamma_pnl,
            'VegaPnL': vega_pnl,
            'IV': sigma_prev,
            'B': B_prev,
            'D': D_prev,
            'OptionPrice': next_value,
            'Underlying': next_price,
            'LogReturn': daily_log_return,
            'VolLogReturn': daily_vol_log_return,
            'Delta': prev_delta,
            'Gamma': prev_gamma,
            'Vega': prev_vega
        })

        prev_price = next_price
        prev_sigma = sigma_prev
        prev_delta, prev_gamma, prev_vega = delta, gamma, vega
        prev_value = next_value

    return pd.DataFrame(all_results)

def main():
    print("Reading data...")
    df = pd.read_pickle("IV_B_D_SPX_data.pkl")
    option_params = pd.read_csv('input_data.csv')

    asian_options = option_params[option_params['OptionType'] == 'Asian']

    if asian_options.empty:
        print("No Asian options found in the input data.")
        return

    df['DataDate'] = pd.to_datetime(df['DataDate'], format='mixed')

    years = df['DataDate'].dt.year.unique()

    for idx, option in asian_options.iterrows():
        print(f"\nProcessing Asian option")
        print(f"Option details: Strike={option['Strike']}, Type={option['OptionType']}")

        single_option_params = pd.DataFrame([option])

        for year in tqdm(years, desc=f"Processing years for strike {option['Strike']}"):
            df_year = df[df['DataDate'].dt.year == year]
            months = df_year['DataDate'].dt.to_period('M').unique()

            for month in tqdm(months, desc=f"Processing months for year {year}", leave=False):
                month_start = month.to_timestamp()
                month_end = month.to_timestamp(how='end')
                df_month = df[(df['DataDate'] >= month_start) & (df['DataDate'] <= month_end)].copy()

                if not df_month.empty:
                    pnl_results = calculate_pnl(df_month, single_option_params)

                    if not pnl_results.empty:
                        year_dir = f"pnl_results_{year}"
                        if not os.path.exists(year_dir):
                            os.makedirs(year_dir)

                        filename = f"{year_dir}/pnl_series_asian_strike_{option['Strike']}_{month}.csv"
                        pnl_results.to_csv(filename)
                        print(f"Saved results for {month} in {year_dir}")

if __name__ == "__main__":
    main()