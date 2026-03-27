import pandas as pd
import numpy as np


def engineer_account_age(df, date_col='account_open_date', reference_date=None):
    """
    Create account_age_days from account_open_date.
        
    Returns:
        Series: days since account open
    """
    if reference_date is None:
        reference_date = pd.Timestamp("2024-01-01")
    
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    return (reference_date - df_temp[date_col]).dt.days


def engineer_financial_ratios(df, balance_col='account_balance', income_col='annual_income_cleaned'):
    """
    Create balance_income_ratio.
    
        
    Returns:
        Series: ratio of balance to income
    """
    eps = 1e-6
    return df[balance_col] / (df[income_col] + eps)


def engineer_engagement_ratios(df, txn_col='monthly_transactions', product_col='num_products',
                        complaint_col='complaints_12mo'):
    """
    Create transaction and engagement ratios.
    
        
    Returns:
        DataFrame with new columns: txn_per_product, complaint_txn_ratio
    """
    df_temp = df.copy()
    df_temp['txn_per_product'] = df_temp[txn_col] / (df_temp[product_col] + 1.0)
    df_temp['complaint_txn_ratio'] = df_temp[complaint_col] / (df_temp[txn_col] + 1.0)
    
    return df_temp[['txn_per_product', 'complaint_txn_ratio']]


def engineer_engagement_flags(df, digital_col='digital_engagement', threshold=40):
    """
    Create low_engagement_flag.
        
    Returns:
        Series: binary flag (1 if engagement < threshold)
    """
    return (df[digital_col] < threshold).astype(int)