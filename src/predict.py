import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def load_model(model_path='models/best_churn_model.pkl'):
    """
    Load serialized pipeline.
    
    Args:
        model_path: Path to pickled model
        
    Returns:
        sklearn Pipeline
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)


def predict_proba_batch(model, X, batch_size=1000):
    """
    Score a batch of customers (memory-efficient).
    
    Args:
        model: Fitted pipeline
        X: Features (DataFrame or array)
        batch_size: Process this many at a time
        
    Returns:
        np.array: churn probabilities [0, 1]
    """
    n = len(X)
    proba = np.zeros(n)
    
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        X_batch = X.iloc[i:end] if isinstance(X, pd.DataFrame) else X[i:end]
        proba[i:end] = model.predict_proba(X_batch)[:, 1]
    
    return proba


def score_and_rank(model, X, customer_ids=None, threshold=0.34):
    """
    Score customers and rank by churn probability.
    
    Args:
        model: Fitted pipeline
        X: Feature matrix
        customer_ids: Optional customer identifiers
        threshold: Binary decision threshold
        
    Returns:
        DataFrame with customer_id, churn_probability, recommended_action, rank
    """
    proba = predict_proba_batch(model, X)
    
    # Create output DataFrame
    results = pd.DataFrame({
        'churn_probability': proba,
        'decision': (proba >= threshold).astype(int),  # 1 = flag for intervention
        'rank': pd.Series(proba).rank(ascending=False, method='dense').values
    })
    
    if customer_ids is not None:
        results.insert(0, 'customer_id', customer_ids)
    
    # Add action recommendation
    results['recommended_action'] = results['decision'].map({
        1: 'Outreach - high churn risk',
        0: 'Monitor - standard engagement'
    })
    
    # Sort by churn probability descending (highest risk first)
    results = results.sort_values('churn_probability', ascending=False).reset_index(drop=True)
    
    return results


def export_for_retention_team(results_df, output_path='retention_targets.csv', top_n=200):
    """
    Export top N flagged customers for retention team.
    
    Args:
        results_df: Output from score_and_rank
        output_path: Where to save CSV
        top_n: Export only top N by churn probability
        
    Returns:
        DataFrame that was exported
    """
    export_df = results_df[results_df['decision'] == 1].head(top_n).copy()
    
    export_df['engagement_priority'] = pd.cut(
        export_df['churn_probability'],
        bins=[0, 0.4, 0.6, 1.0],
        labels=['Standard', 'High', 'Urgent']
    )
    
    export_df.to_csv(output_path, index=False)
    print(f"Exported {len(export_df)} customers to {output_path}")
    
    return export_df


def estimate_intervention_cost(results_df, cost_per_call=5, cost_per_lost_customer=500):
    """
    Estimate cost/benefit of retention campaign.
    
    Args:
        results_df: Scores dataframe
        cost_per_call: $ per outreach attempt
        cost_per_lost_customer: $ lost per churned customer
        
    Returns:
        dict with cost metrics
    """
    flagged = (results_df['decision'] == 1).sum()
    
    # Assume flag rate correlates with actual churn (simplified)
    estimated_churners_among_flagged = (results_df[results_df['decision'] == 1]['churn_probability'].mean() 
                                        * flagged)
    
    campaign_cost = flagged * cost_per_call
    expected_savings = estimated_churners_among_flagged * cost_per_lost_customer
    net_benefit = expected_savings - campaign_cost
    
    return {
        'customers_flagged': int(flagged),
        'estimated_churners_if_notreated': int(estimated_churners_among_flagged),
        'campaign_cost': campaign_cost,
        'expected_savings': expected_savings,
        'net_benefit': net_benefit,
        'roi': net_benefit / max(campaign_cost, 1)  # Avoid division by zero
    }