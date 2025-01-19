import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ks_2samp, spearmanr
import os
pd.set_option('display.max_columns', None)

def process_pnl_files(base_dir):
    # Process European option data (single file in base directory)
    european_df = pd.read_csv(os.path.join(base_dir, 'combined_european_pnl.csv'))

    # Initialize empty lists for Asian and Lookback dataframes
    asian_dfs = []
    lookback_dfs = []

    # Process files from each year folder (2019-2024)
    for year in range(2019, 2025):
        year_folder = f"pnl_results_{year}"
        year_path = os.path.join(base_dir, year_folder)

        if os.path.exists(year_path):
            # Find Asian and Lookback files in the year folder
            asian_files = glob(os.path.join(year_path, "pnl_series_asian_strike_*.csv"))
            lookback_files = glob(os.path.join(year_path, "pnl_series_lookback_strike_*.csv"))

            # Process Asian files for this year
            for file in asian_files:
                df = pd.read_csv(file)
                asian_dfs.append(df)

            # Process Lookback files for this year
            for file in lookback_files:
                df = pd.read_csv(file)
                lookback_dfs.append(df)

    # Combine all Asian and Lookback files
    asian_combined = pd.concat(asian_dfs, ignore_index=True) if asian_dfs else pd.DataFrame()
    lookback_combined = pd.concat(lookback_dfs, ignore_index=True) if lookback_dfs else pd.DataFrame()

    # Save combined files
    asian_combined.to_csv('combined_asian_pnl.csv', index=False)
    lookback_combined.to_csv('combined_lookback_pnl.csv', index=False)

    # PnL columns to keep in final file
    pnl_columns = ['DataDate', 'FullRevalPnL', 'TheoreticalPnL', 'UnexplainedPnL',
                   'DeltaPnL', 'GammaPnL', 'VegaPnL']

    # Filter and rename columns for each option type
    european_subset = european_df[pnl_columns].copy()
    european_subset.columns = ['DataDate'] + [f'European_{col}' for col in pnl_columns[1:]]

    asian_subset = asian_combined[pnl_columns].copy() if not asian_combined.empty else pd.DataFrame(columns=pnl_columns)
    asian_subset.columns = ['DataDate'] + [f'Asian_{col}' for col in pnl_columns[1:]]

    lookback_subset = lookback_combined[pnl_columns].copy() if not lookback_combined.empty else pd.DataFrame(
        columns=pnl_columns)
    lookback_subset.columns = ['DataDate'] + [f'Lookback_{col}' for col in pnl_columns[1:]]

    # Merge all options data on DataDate
    final_df = european_subset.merge(asian_subset, on='DataDate', how='outer') \
        .merge(lookback_subset, on='DataDate', how='outer')

    # Sort by date
    final_df['DataDate'] = pd.to_datetime(final_df['DataDate'])
    final_df.sort_values('DataDate', inplace=True)

    # Add combined PnL columns
    pnl_types = ['FullRevalPnL', 'TheoreticalPnL', 'UnexplainedPnL', 'DeltaPnL', 'GammaPnL', 'VegaPnL']
    for pnl_type in pnl_types:
        final_df[pnl_type] = final_df[f'European_{pnl_type}'] + \
                             final_df[f'Asian_{pnl_type}'] + \
                             final_df[f'Lookback_{pnl_type}']

    final_df = final_df[abs(final_df['FullRevalPnL']) > 1e-10]

    # Save final combined file
    final_df.to_csv('combined_all_options_pnl.csv', index=False)

    return final_df

def plot_pnls(pnl_results):
    """Plot PnL results for all options combined with unexplained PnL subplot"""
    if pnl_results.empty:
        print("No PnL results to plot!")
        return

    pnl_results['DataDate'] = pd.to_datetime(pnl_results['DataDate'])

    # Create figure with 2 subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})

    # Top subplot for Full Revaluation and Theoretical PnL
    ax1.plot(pnl_results['DataDate'], pnl_results['FullRevalPnL'],
             label='Full Revaluation PnL', color='blue')
    ax1.plot(pnl_results['DataDate'], pnl_results['TheoreticalPnL'],
             label='Theoretical PnL', color='red', linestyle='--')
    ax1.set_title('Combined Options PnL Over Time')
    ax1.set_ylabel('PnL')
    ax1.legend()
    ax1.grid(True)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Bottom subplot for Unexplained PnL
    ax2.plot(pnl_results['DataDate'], pnl_results['UnexplainedPnL'],
             label='Unexplained PnL', color='green')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Unexplained PnL')
    ax2.legend()
    ax2.grid(True)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def calculate_metrics(pnl_results):
    """Calculate and display metrics for each year from 2019 to 2024"""
    # Convert DataDate to datetime if it's not already
    pnl_results['DataDate'] = pd.to_datetime(pnl_results['DataDate'])

    # Create empty list to store metrics for each year
    yearly_metrics = []

    # Calculate metrics for each year
    for year in range(2019, 2025):
        # Filter data for current year
        year_data = pnl_results[pnl_results['DataDate'].dt.year == year].copy()

        if not year_data.empty:
            columns_to_check = ['FullRevalPnL', 'TheoreticalPnL']
            year_data_cleaned = year_data.dropna(subset=columns_to_check)
            year_data_cleaned = year_data_cleaned[~(year_data_cleaned[columns_to_check] == 0).any(axis=1)]

            if len(year_data_cleaned) > 1:  # Need at least 2 points for the statistical tests
                ks_stat, ks_pvalue = ks_2samp(year_data_cleaned['FullRevalPnL'],
                                              year_data_cleaned['TheoreticalPnL'],
                                              method='asymp')
                spearman_corr, spearman_pvalue = spearmanr(year_data_cleaned['FullRevalPnL'],
                                                           year_data_cleaned['TheoreticalPnL'])

                metrics = {
                    'Year': year,
                    'KS_Statistic': ks_stat,
                    'KS_P_Value': ks_pvalue,
                    'Spearman_Correlation': spearman_corr,
                    'Spearman_P_Value': spearman_pvalue,
                    'Mean_Unexplained_PnL': year_data_cleaned['UnexplainedPnL'].mean(),
                    'Std_Unexplained_PnL': year_data_cleaned['UnexplainedPnL'].std(),
                    'Number_of_Observations': len(year_data_cleaned)
                }

                yearly_metrics.append(metrics)

                print(f'\nMetrics for {year}:')
                print(f'KS Statistic: {ks_stat:.4f}')
                print(f'Spearman Correlation: {spearman_corr:.4f}')
                print(f'Number of Observations: {len(year_data_cleaned)}')
            else:
                print(f'\nInsufficient data for year {year} to calculate metrics')
        else:
            print(f'\nNo data available for year {year}')

    # Create DataFrame with all yearly metrics
    metrics_df = pd.DataFrame(yearly_metrics)

    # Set Year as index for better readability
    metrics_df.set_index('Year', inplace=True)

    return metrics_df


def var_backtest(pnl_results, confidence_level=0.99, window_size=252):
    """
    Perform VaR backtesting using historical simulation method

    Parameters:
    pnl_results: DataFrame with DataDate and TheoreticalPnL columns
    confidence_level: Confidence level for VaR calculation (default 99%)
    window_size: Rolling window size in days (default 252 - one trading year)

    Returns:
    tuple: (violations_ratio, var_series, actual_pnl_series)
    """
    # Convert DataDate to datetime if not already
    pnl_results['DataDate'] = pd.to_datetime(pnl_results['DataDate'])

    # Split data into training (up to June 2023) and testing (July 2023 onwards)
    train_data = pnl_results[pnl_results['DataDate'] < '2023-07-01'].copy()
    test_data = pnl_results[pnl_results['DataDate'] >= '2023-07-01'].copy()

    # Initialize series to store VaR values
    var_series = pd.Series(index=test_data.index, dtype=float)

    # Calculate VaR for each day in the test period
    for i in range(len(test_data)):
        current_date = test_data.iloc[i]['DataDate']

        # Get historical window end date (day before current date)
        window_end = current_date - pd.Timedelta(days=1)

        # Get historical window data
        historical_window = train_data[
            train_data['DataDate'] <= window_end
            ].tail(window_size)['TheoreticalPnL']

        # Calculate VaR using historical simulation
        var = historical_window.quantile(1 - confidence_level)
        var_series.iloc[i] = var

    # Count VaR violations
    actual_pnl = test_data['TheoreticalPnL']
    violations = (actual_pnl < var_series).sum()
    violations_ratio = violations / len(test_data)

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_data['DataDate'], actual_pnl, label='Actual PnL', color='blue')
    plt.plot(test_data['DataDate'], var_series, label=f'{confidence_level * 100}% VaR',
             color='red', linestyle='--')
    plt.fill_between(test_data['DataDate'], var_series, min(var_series.min(), actual_pnl.min()),
                     color='red', alpha=0.1)

    plt.title('VaR Backtesting Results')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Print backtesting results
    expected_violations = (1 - confidence_level) * len(test_data)
    print(f"\nBacktesting Results:")
    print(f"Number of observations: {len(test_data)}")
    print(f"Number of violations: {violations}")
    print(f"Expected number of violations: {expected_violations:.2f}")
    print(f"Violations ratio: {violations_ratio:.4f}")
    print(f"Expected violations ratio: {1 - confidence_level:.4f}")

    return violations_ratio, var_series, actual_pnl

# Example usage
base_dir = "C:/Users/aniru/PycharmProjects/Exotics_Risk_Pricing"

final_df = process_pnl_files(base_dir)
plot_pnls(final_df)
yearly_metrics_df = calculate_metrics(final_df)
print("\nComplete yearly metrics:")
print(yearly_metrics_df)
var_backtest(final_df)
