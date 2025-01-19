import pandas as pd
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.stats import norm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import os
pd.set_option('display.max_columns', None)


def calculate_year_fraction(start_date, end_date):
    delta = end_date - start_date
    total_days = delta.days

    leap_days = 0
    for year in range(start_date.year, end_date.year + 1):
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            leap_date = datetime(year, 2, 29)
            if start_date < leap_date < end_date:
                leap_days += 1

    years = relativedelta(end_date, start_date).years
    remaining_days = total_days - (years * 365 + leap_days)

    return years + remaining_days / 365.0

def objective(x, K, C, P, S0):
    B, D = x
    return np.sum((C - P - D * S0 + B * K) ** 2)

def black_scholes(S, K, sigma, B, D, T, option_type):
    d1 = (np.log(S * D / (K * B)) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * D * norm.cdf(d1) - K * B * norm.cdf(d2)
    else:
        return K * B * norm.cdf(-d2) - S * D * norm.cdf(-d1)


def black_scholes_imp_vol(S0, K, B, D, T, option_type, market_price):
    def objective_function(sigma):
        model_price = black_scholes(S0, K, sigma, B, D, T, option_type)
        return model_price - market_price

    sol = root_scalar(objective_function, bracket=[0.00001, 5], method='brentq')
    return sol.root

def process_data(df):
    results = []

    date_pairs = df.groupby(['DataDate', 'ExpirationDate']).size().reset_index()
    total_combinations = len(date_pairs)

    with tqdm(total=total_combinations, desc="Processing options data") as pbar:
        for (data_date, expiry_date), group in df.groupby(['DataDate', 'ExpirationDate']):
            try:
                data_date_dt = pd.to_datetime(data_date)
                expiry_date_dt = pd.to_datetime(expiry_date)
                T = calculate_year_fraction(data_date_dt, expiry_date_dt)

                K = group['StrikePrice'].values
                C = group['CallMidPrice'].values
                P = group['PutMidPrice'].values
                S0 = group['UnderlyingPrice'].iloc[0]

                # Calculate B & D
                x0 = [1.0, 1.0]
                result = minimize(objective, x0, args=(K, C, P, S0),
                                  method='Nelder-Mead',
                                  options={'maxiter': 1000})

                B, D = result.x

                # Calculate implied vol for each strike in this group
                for idx, row in group.iterrows():
                    try:
                        imp_vol = black_scholes_imp_vol(
                            S0=S0,
                            K=row['StrikePrice'],
                            B=B,
                            D=D,
                            T=T,
                            option_type='call',
                            market_price=row['CallMidPrice']
                        )
                    except:
                        imp_vol = np.nan

                    results.append({
                        'DataDate': data_date,
                        'ExpirationDate': expiry_date,
                        'UnderlyingPrice': S0,
                        'StrikePrice': row['StrikePrice'],
                        'T': T,
                        'B': B,
                        'D': D,
                        'ImpliedVol': imp_vol
                    })

            except Exception as e:
                # If B&D calculation fails, add entry for each strike with NaN values
                for idx, row in group.iterrows():
                    results.append({
                        'DataDate': data_date,
                        'ExpirationDate': expiry_date,
                        'UnderlyingPrice': S0,
                        'StrikePrice': row['StrikePrice'],
                        'T': T,
                        'B': np.nan,
                        'D': np.nan,
                        'ImpliedVol': np.nan
                    })

            pbar.update(1)

    return pd.DataFrame(results)

comibned_df = pd.read_pickle("combined_SPX_data_1.pkl")
# Process the data
results_df = process_data(comibned_df)

# Sort the results
results_df = results_df.sort_values(['DataDate', 'ExpirationDate', 'StrikePrice']).reset_index(drop=True)
#Stroe the results in pickle file
pickle_path = os.path.join(r"C:\Users\aniru\PycharmProjects\Exotics_Risk_Pricing", "IV_B_D_SPX_data_1.pkl")
results_df.to_pickle(pickle_path)
results_df.to_csv('results_df.csv')
# # Merge with original dataframe if needed
# final_df = comibned_df.merge(results_df, on=['DataDate', 'ExpirationDate', 'StrikePrice'], how='left')