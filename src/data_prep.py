
import re
import pandas as pd
import numpy as np


def parse_income(val):
    """
    Returns:
        float or NaN if unparseable
    """
    # Already numeric
    if isinstance(val, (int, float)):
        return float(val)
    
    # Missing value
    if pd.isna(val):
        return np.nan
    
    # String cleanup
    s = str(val).strip().upper()
    s = re.sub(r'[PHP\$,\s]', '', s)  # remove currency, commas, spaces
    
    # Handle K(thousands) / M(millions) suffix
    multiplier = 1
    if s.endswith('K'):
        s = s[:-1]
        multiplier = 1000
    elif s.endswith('M'):
        s = s[:-1]
        multiplier = 1_000_000
    
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def check_for_leakage(df, target='churned', leak_cols=None):
    """
        
    Returns:
        dict with leak diagnosis
    """
    if leak_cols is None:
        leak_cols = ['exit_survey_score', 'account_closed_date', 'churn_flag_internal']
    
    results = {}
    for col in leak_cols:
        if col not in df.columns:
            continue
        
        not_null_rate = df[col].notna().mean()
        if not_null_rate == 0:
            continue
            
        not_null_if_churn = df.loc[df[target] == 1, col].notna().mean()
        not_null_if_not = df.loc[df[target] == 0, col].notna().mean()
        
        results[col] = {
            'not_null_rate': not_null_rate,
            'not_null_if_churn': not_null_if_churn,
            'not_null_if_not_churn': not_null_if_not,
            'is_suspicious': abs(not_null_if_churn - not_null_if_not) > 0.5
        }
    
    return results


def check_duplicates_by_id(df, id_col='customer_id'):
    """
    Returns:
        int: number of duplicate IDs
    """
    return df[id_col].duplicated().sum()