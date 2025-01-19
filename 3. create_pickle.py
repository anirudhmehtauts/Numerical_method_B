import pandas as pd
import glob
import os

# Base directory path
base_dir = r"C:\Users\aniru\PycharmProjects\Exotics_Risk_Pricing\Market_Data_SPX_1"


def extract_date(filename):
    """Extract date from different filename formats"""
    basename = os.path.basename(filename)
    if basename.startswith('D_'):
        # For format D_20240101_OData_SPX
        return basename.split('_')[1]
    else:
        # For format 20050103_OData_SPX
        return basename.split('_')[0]


def combine_data_files(base_path):
    dfs = []

    year_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

    for year in year_folders:
        year_path = os.path.join(base_path, year)
        # Pattern for both file formats
        file_patterns = [
            os.path.join(year_path, "D_*_OData_SPX.csv"),  # New format
            os.path.join(year_path, "[0-9]*_OData_SPX.csv")  # Old format
        ]

        for pattern in file_patterns:
            for file in sorted(glob.glob(pattern)):
                try:
                    df = pd.read_csv(file)
                    # Add date from filename using the new extract_date function
                    date_str = extract_date(file)
                    df['file_date'] = pd.to_datetime(date_str)

                    # Create separate dataframes for puts and calls
                    df_calls = df[df['PutCall'] == 'call'][
                        ['Symbol', 'ExpirationDate', 'AskPrice', 'BidPrice', 'StrikePrice','UnderlyingPrice','DataDate']]
                    df_puts = df[df['PutCall'] == 'put'][
                        ['Symbol', 'ExpirationDate', 'AskPrice', 'BidPrice', 'StrikePrice']]

                    # Rename columns
                    df_calls = df_calls.rename(columns={'AskPrice': 'CallAskPrice'})
                    df_calls = df_calls.rename(columns={'BidPrice': 'CallBidPrice'})
                    df_puts = df_puts.rename(columns={'AskPrice': 'PutAskPrice'})
                    df_puts = df_puts.rename(columns={'BidPrice': 'PutBidPrice'})

                    # Merge the dataframes
                    df = pd.merge(
                        df_calls,
                        df_puts,
                        on=['Symbol', 'ExpirationDate', 'StrikePrice'],
                        how='outer'
                    )

                    # Add PutCall.1 column
                    df['PutCall.1'] = 'put'

                    # Calculate mid prices
                    df['CallMidPrice'] = (df['CallBidPrice'] + df['CallAskPrice']) / 2
                    df['PutMidPrice'] = (df['PutBidPrice'] + df['PutAskPrice']) / 2

                    dfs.append(df)
                    print(f"Processed: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    return dfs


# Process all files
dfs = combine_data_files(base_dir)

# Combine all dataframes if any were processed
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to pickle file
    pickle_path = os.path.join(r"C:\Users\aniru\PycharmProjects\Exotics_Risk_Pricing", "combined_SPX_data_1.pkl")
    pd.set_option('display.max_columns', None)
    combined_df.to_pickle(pickle_path)
    print(combined_df)
    print(f"\nSuccessfully created pickle file at: {pickle_path}")
    print(f"Combined shape: {combined_df.shape}")
else:
    print("No files were processed")