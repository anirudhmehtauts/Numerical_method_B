import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ks_2samp, spearmanr
from tqdm import tqdm
pd.set_option('display.max_columns', None)


def interpolate_surface(df_date, T, K, param):
    if df_date.empty:
        return None

    # Clean data by removing NaN values for this parameter
    df_clean = df_date[df_date[param].notna()]
    if df_clean.empty:
        return None

    # Get unique T values and find nearest neighbors
    unique_T = sorted(df_clean['T'].unique())
    if len(unique_T) < 2:
        return None

    T_idx = np.searchsorted(unique_T, T)
    T_idx = min(max(1, T_idx), len(unique_T) - 1)
    T_lower, T_upper = unique_T[T_idx - 1], unique_T[T_idx]

    # Get slices for lower and upper T and handle duplicates
    lower_slice = (df_clean[df_clean['T'] == T_lower]
                  .drop_duplicates('StrikePrice')
                  .sort_values('StrikePrice'))
    upper_slice = (df_clean[df_clean['T'] == T_upper]
                  .drop_duplicates('StrikePrice')
                  .sort_values('StrikePrice'))

    if len(lower_slice) < 4 or len(upper_slice) < 4:
        return None

    # Check if K is outside the available strike range
    lower_min_strike = max(lower_slice['StrikePrice'].min(), upper_slice['StrikePrice'].min())
    lower_max_strike = min(lower_slice['StrikePrice'].max(), upper_slice['StrikePrice'].max())

    if K < lower_min_strike:
        # Return interpolated IV for the lowest available strike
        return interpolate_at_strike(lower_slice, upper_slice, T, T_lower, T_upper, lower_min_strike, param)
    elif K > lower_max_strike:
        # Return interpolated IV for the highest available strike
        return interpolate_at_strike(lower_slice, upper_slice, T, T_lower, T_upper, lower_max_strike, param)

    try:
        # Regular interpolation for strikes within range
        cs_lower = CubicSpline(lower_slice['StrikePrice'], lower_slice[param])
        cs_upper = CubicSpline(upper_slice['StrikePrice'], upper_slice[param])

        val_lower = cs_lower(K)
        val_upper = cs_upper(K)

        result = val_lower + (val_upper - val_lower) * (T - T_lower) / (T_upper - T_lower)
        return result if np.isfinite(result) else None

    except Exception as e:
        print(f"Interpolation error for {param}: {e}")
        return None

def interpolate_at_strike(lower_slice, upper_slice, T, T_lower, T_upper, K, param):
    """Helper function to interpolate at a specific strike across time"""
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

def black_scholes(S, K, sigma, B, D, T, option_type):
    d1 = (np.log(S * D / (K * B)) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * D * norm.cdf(d1) - K * B * norm.cdf(d2)
    else:
        return K * B * norm.cdf(-d2) - S * D * norm.cdf(-d1)
def calculate_greeks(S, K, sigma, B, D, T):
    """Calculate Delta, Gamma, and Vega"""
    if T <= 0:
        return 0, 0, 0

    d1 = (np.log(S * D / (K * B)) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = D * norm.cdf(d1)
    gamma = (D * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * D * np.sqrt(T) * norm.pdf(d1)

    return delta, gamma, vega


def calculate_pnl(df, option_params):
    df['DataDate'] = pd.to_datetime(df['DataDate'], format='mixed').dt.strftime('%Y-%m-%d')
    df = df.sort_values(['DataDate', 'T', 'StrikePrice'])

    # Get the reference date and data
    ref_date = df['DataDate'].max()
    ref_data = df[df['DataDate'] == ref_date]

    # Get all historical dates before the reference date
    dates = sorted(df[df['DataDate'] < ref_date]['DataDate'].unique(), reverse=True)

    option_maturity = option_params['Maturity(years)'].iloc[0]
    option_type = option_params['CallPut'].iloc[0].lower()
    strike = option_params['Strike'].iloc[0]

    # Calculate reference values (not used for PnL but as initialization)
    current_price = ref_data['UnderlyingPrice'].iloc[0]
    B_current = interpolate_surface(ref_data, option_maturity, strike, 'B')
    D_current = interpolate_surface(ref_data, option_maturity, strike, 'D')
    sigma_current = interpolate_surface(ref_data, option_maturity, strike, 'ImpliedVol')

    if any(v is None for v in [B_current, D_current, sigma_current]):
        return pd.DataFrame()

    current_value = black_scholes(current_price, strike, sigma_current, B_current, D_current,
                                  option_maturity, option_type)

    # Initialize previous values for historical dates
    prev_price = current_price
    prev_sigma = sigma_current
    prev_delta, prev_gamma, prev_vega = calculate_greeks(current_price, strike, sigma_current,
                                                         B_current, D_current, option_maturity)
    prev_value = current_value

    results = []

    # Loop through historical dates and calculate PnL
    for date in tqdm(dates, desc="Calculating PnL"):
        next_data = df[df['DataDate'] == date]
        next_price = next_data['UnderlyingPrice'].iloc[0]

        T_prev = option_maturity
        B_prev = interpolate_surface(next_data, T_prev, strike, 'B')
        D_prev = interpolate_surface(next_data, T_prev, strike, 'D')
        sigma_prev = interpolate_surface(next_data, T_prev, strike, 'ImpliedVol')

        # Skip if all three (B, D, IV) are zero
        if all(v < 0.0001 for v in [B_prev, D_prev, sigma_prev]):
            print(f"Skipping date {date} due to data issues (B, D, IV all zero).")
            continue

        if any(v is None for v in [B_prev, D_prev, sigma_prev]):
            continue

        # Calculate Greeks for the current date
        delta, gamma, vega = calculate_greeks(next_price, strike, sigma_prev, B_prev, D_prev, T_prev)

        # Full revaluation based on next and current values
        next_value = black_scholes(next_price, strike, sigma_prev, B_prev, D_prev, T_prev, option_type)
        full_reval_pnl = next_value - prev_value

        # Calculate PnL components using Greeks of the previous date
        daily_log_return = np.log(next_price / prev_price)
        daily_vol_log_return = np.log(sigma_prev / prev_sigma)

        delta_pnl = daily_log_return * prev_price * prev_delta
        gamma_pnl = (0.5 * prev_gamma * (prev_price * daily_log_return) ** 2
                     + 0.5 * prev_delta * prev_price * (daily_log_return ** 2))


        vega_pnl = daily_vol_log_return * prev_vega * prev_sigma

        theoretical_pnl = delta_pnl + gamma_pnl + vega_pnl
        unexplained_pnl = abs(theoretical_pnl-full_reval_pnl)

        results.append({
            'DataDate': date,
            'FullRevalPnL': full_reval_pnl,
            'TheoreticalPnL': theoretical_pnl,
            'UnexplainedPnL': unexplained_pnl,
            'DeltaPnL': delta_pnl,
            'GammaPnL': gamma_pnl,
            'VegaPnL': vega_pnl,
            'IV': sigma_prev,
            'B': B_prev,
            'D': D_prev,
            'Current_opt_price': current_value,
            'Previous_opt_Price': next_value,
            'Underlying': next_price,
            'LogReturn': daily_log_return,
            'VolLogReturn': daily_vol_log_return,
            'Delta': prev_delta,
            'Gamma': prev_gamma,
            'Vega': prev_vega
        })

        # Update previous values for next iteration
        prev_price = next_price
        prev_sigma = sigma_prev
        prev_delta, prev_gamma, prev_vega = delta, gamma, vega
        prev_value = next_value

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        columns_to_check = ['FullRevalPnL', 'TheoreticalPnL']
        results_df_cleaned = results_df.dropna(subset=columns_to_check)
        results_df_cleaned = results_df_cleaned[~(results_df_cleaned[columns_to_check] == 0).any(axis=1)]

        ks_stat, ks_pvalue = ks_2samp(results_df_cleaned['FullRevalPnL'], results_df_cleaned['TheoreticalPnL'])
        spearman_corr, spearman_pvalue = spearmanr(results_df_cleaned['FullRevalPnL'], results_df_cleaned['TheoreticalPnL'])
        print(f"KS Test Statistic: {ks_stat}, p-value: {ks_pvalue}")
        print(f"Spearman Correlation: {spearman_corr}, p-value: {spearman_pvalue}")

        # Add metrics to results
        results_df_cleaned['KS_Test_Stat'] = ks_stat
        results_df_cleaned['Spearman_Corr'] = spearman_corr

    return results_df_cleaned

# Read the option parameters
def read_option_params(filepath):
    return pd.read_csv(filepath)


def plot_pnls(pnl_results):
    if pnl_results.empty:
        print("No PnL results to plot!")
        return

    # Convert DataDate to datetime without specifying format
    # pandas will automatically detect the ISO format YYYY-MM-DD
    pnl_results['DataDate'] = pd.to_datetime(pnl_results['DataDate'])

    # Rest of the function remains the same
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_results['DataDate'], pnl_results['FullRevalPnL'],
             label='Full Revaluation PnL', color='blue')
    plt.plot(pnl_results['DataDate'], pnl_results['TheoreticalPnL'],
             label='Theoretical PnL', color='red', linestyle='--')

    plt.title('Full Revaluation vs Theoretical PnL Over Time')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

df = pd.read_pickle("IV_B_D_SPX_data.pkl")
# Main execution
option_params = read_option_params('input_data.csv')
pnl_results = calculate_pnl(df, option_params)
pnl_results.to_csv('combined_european_pnl.csv')
# Plot results
plot_pnls(pnl_results)