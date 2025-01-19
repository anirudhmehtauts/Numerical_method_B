import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# If dates are in column 'DataDate' in format 'dd/mm/yyyy'
dates_df = pd.read_csv('pnl_series.csv', parse_dates=['DataDate'], dayfirst=False)
# Convert dates from string to datetime
dates = pd.to_datetime(dates_df['DataDate'], format='mixed')


def generate_correlated_spot_vol(spot_change, corr=-0.3):
    # Incorporate correlation between spot and vol moves
    vol_change = corr * spot_change + np.sqrt(1 - corr ** 2) * np.random.normal()
    return vol_change


def check_calendar_arbitrage(vols, maturities):
    for i in range(len(maturities) - 1):
        if vols[i] > vols[i + 1]:
            # Adjust to prevent calendar arbitrage
            vols[i + 1] = vols[i] * 1.001
    return vols


def generate_historical_spot_and_vol():
    np.random.seed(42)
    spot_data = []
    vol_data = []  # To store base volatilities
    initial_spot = 1.20
    initial_vol = 0.08

    annual_drift = 0.001
    annual_vol = 0.08
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    current_spot = initial_spot
    current_vol = initial_vol

    for _ in range(len(dates)):
        # Generate spot change
        spot_change = daily_drift - 0.5 * daily_vol ** 2 + daily_vol * np.random.normal()
        current_spot *= np.exp(spot_change)

        # Generate correlated vol change
        vol_change = generate_correlated_spot_vol(spot_change)
        current_vol = max(0.01, current_vol * (1 + vol_change))  # Ensure vol stays positive

        spot_data.append(current_spot)
        vol_data.append(current_vol)

    return np.array(spot_data), np.array(vol_data)


def generate_fx_vol_smile(spot, strike_range, center_vol, smile_factor, days_to_expiry):
    strikes = np.linspace(spot * (1 - strike_range), spot * (1 + strike_range), 11)

    # Create smile effect with term structure
    moneyness = (strikes - spot) / spot
    term_adjustment = np.sqrt(days_to_expiry / 365) * 0.02
    vols = center_vol + term_adjustment + smile_factor * moneyness ** 2 + smile_factor * moneyness * 0.1

    return strikes, vols


def create_dataset():
    spot_prices, base_vols = generate_historical_spot_and_vol()

    # Parameters for discount and dividend curves
    risk_free_rate = 0.03  # Annualized risk-free rate
    dividend_yield = 0.02  # Annualized dividend yield

    data = []
    maturities = [30, 90, 180, 270, 365, 456, 547, 638, 730]
    maturity_labels = ['1M', '3M', '6M', '9M', '1Y', '1.25Y', '1.5Y', '1.75Y', '2Y']

    for date, spot, base_vol in zip(dates, spot_prices, base_vols):
        for days, label in zip(maturities, maturity_labels):
            maturity_date = date + timedelta(days=days)
            time_to_maturity = days / 365

            # Compute discount and dividend factors
            discount_factor = np.exp(-risk_free_rate * time_to_maturity)
            dividend_factor = np.exp(-dividend_yield * time_to_maturity)

            smile_factor = 0.15 * np.exp(-0.2 * days / 365)

            strikes, vols = generate_fx_vol_smile(
                spot=spot,
                strike_range=0.5,
                center_vol=base_vol,
                smile_factor=smile_factor,
                days_to_expiry=days
            )

            # Check and adjust for calendar spread arbitrage
            term_structure_vols = [v.mean() for v in vols]  # Use average vol for each maturity
            adjusted_term_vols = check_calendar_arbitrage(term_structure_vols, maturities)

            # Adjust individual smile vols based on term structure correction
            vol_adjustment = adjusted_term_vols[maturities.index(days)] / term_structure_vols[maturities.index(days)]
            vols = vols * vol_adjustment

            for strike, vol in zip(strikes, vols):
                data.append({
                    'DataDate': date,
                    'ExpirationDate': maturity_date,
                    'UnderlyingPrice': spot,
                    'StrikePrice': strike,
                    'ImpliedVol': vol,
                    'T': time_to_maturity,
                    'B': discount_factor,
                    'D': dividend_factor,
                    'Maturity': label
                })

    return pd.DataFrame(data)


# Create and save the dataset
df = create_dataset()

# Save to pickle file
df.to_pickle('fx_volatility_data.pkl')
df.to_csv('fx_volatility_data.csv')

# Display sample of the data
print("\nSample of the generated data:")
print(df.head())

# Print some statistics
print("\nDataset Statistics:")
print(f"Total number of records: {len(df)}")
print(f"Date range: {df['DataDate'].min()} to {df['DataDate'].max()}")
print(f"Spot price range: {df['UnderlyingPrice'].min():.4f} to {df['UnderlyingPrice'].max():.4f}")
print(f"Volatility range: {df['ImpliedVol'].min():.4f} to {df['ImpliedVol'].max():.4f}")

# Verify correlation
spot_returns = df.groupby('DataDate')['UnderlyingPrice'].first().pct_change()
vol_returns = df.groupby('DataDate')['ImpliedVol'].mean().pct_change()
correlation = spot_returns.corr(vol_returns)
print(f"\nSpot-Vol Correlation: {correlation:.3f}")

# Plot spot-strike curves
sample_dates = pd.date_range(start=df['DataDate'].min(),
                             end=df['DataDate'].max(),
                             periods=4)

# Create spot-strike plot
plt.figure(figsize=(12, 8))

for date in sample_dates:
    # Get closest available date in dataset
    closest_date = df['DataDate'].iloc[(df['DataDate'] - date).abs().argsort()[0]]

    # Filter data for this date and the 3M maturity
    date_data = df[(df['DataDate'] == closest_date) & (df['T'] == 0.25)]

    # Plot spot-strike curve
    plt.plot(date_data['StrikePrice'],
             date_data['ImpliedVol'],
             marker='o',
             label=f'Date: {closest_date.strftime("%Y-%m-%d")}')

plt.title('FX Volatility Smile - 3M Maturity')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.legend()
plt.grid(True)
plt.show()

# Create maturity plot
latest_date = df['DataDate'].max()
plt.figure(figsize=(12, 8))

for maturity in ['1M', '3M', '6M', '1Y']:
    maturity_data = df[(df['DataDate'] == latest_date) & (df['Maturity'] == maturity)]
    plt.plot(maturity_data['StrikePrice'],
             maturity_data['ImpliedVol'],
             marker='o',
             label=f'Maturity: {maturity}')

plt.title(f'FX Volatility Smile by Maturity\nDate: {latest_date.strftime("%Y-%m-%d")}')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.legend()
plt.grid(True)
plt.show()