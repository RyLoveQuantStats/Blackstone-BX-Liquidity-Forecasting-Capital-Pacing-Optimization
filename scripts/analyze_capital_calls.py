import pandas as pd
import numpy as np

# Load merged dataset
file_path = 'output/bx_master_data.csv'
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Columns potentially useful for PE capital calls/distributions analysis
relevant_columns = [
    'netCashProvidedByOperatingActivities', 'netCashUsedForInvestingActivites',
    'dividendsPaid', 'purchasesOfInvestments', 'retainedEarnings',
    'commonStockIssued', 'commonStockRepurchased', 'debtRepayment', 'debtIssued'
]

# Filter available columns
available_columns = [col for col in relevant_columns if col in df.columns]

# Create derived metrics commonly used in PE for capital calls:
df['net_financing_activity'] = df.get('debtIssued', 0) - df.get('debtRepayment', 0)
df['net_equity_activity'] = df.get('commonStockIssued', 0) - df.get('commonStockRepurchased', 0)
df['investment_activity'] = df.get('purchasesOfInvestments', 0) - df.get('netCashUsedForInvestingActivites', 0)
df['operational_cashflow'] = df.get('netCashProvidedByOperatingActivities', 0)

# Analyze changes for capital call patterns
capital_call_estimate = df['investment_activity'].apply(lambda x: x if x > 0 else 0)
distribution_estimate = df['net_financing_activity'].apply(lambda x: x if x < 0 else 0) * -1

# Analyze volatility and correlation of these metrics with net cash flows
corr_matrix = df[[
    'net_financing_activity', 'net_equity_activity',
    'investment_activity', 'operational_cashflow'
]].corr()

# Output analysis
print("Available columns for analysis:", available_columns)
print("\nCapital call estimate (top 5):\n", capital_call_estimate.head())
print("\nDistribution estimate (top 5):\n", distribution_estimate.head())
print("\nCorrelation matrix:\n", corr_matrix)

# Save outputs for review
summary_path = 'output/pe_capital_calls_analysis.csv'
df[['net_financing_activity', 'net_equity_activity', 'investment_activity', 'operational_cashflow']].describe().to_csv(summary_path)
print(f"Analysis saved to {summary_path}")

import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

def calculate_capital_calls(df):
    # Calculate net financing inflows (debt issued + equity issued)
    net_financing_inflows = df.get('debtIssued', 0) + df.get('commonStockIssued', 0)

    # Calculate net investment outflows
    investment_outflows = df.get('purchasesOfInvestments', 0)

    # Calculate operational cash flow
    operational_cashflow = df.get('netCashProvidedByOperatingActivities', 0)

    # Calculate shortfall
    shortfall = investment_outflows - (operational_cashflow + net_financing_inflows)

    # Capital call is the positive shortfall (if any)
    df['capital_calls'] = shortfall.apply(lambda x: max(0, x))

    return df

def analyze_results(df):
    print("Capital Calls Analysis:\n")
    print(df['capital_calls'].describe())
    print("\nTop Capital Calls:\n", df['capital_calls'].nlargest(5))

def main():
    file_path = 'output/bx_master_data.csv'
    df = load_data(file_path)
    df = calculate_capital_calls(df)
    analyze_results(df)
    output_path = 'output/pe_capital_calls_estimates.csv'
    df[['capital_calls']].to_csv(output_path)
    print(f"Capital call estimates saved to {output_path}")

if __name__ == "__main__":
    main()
