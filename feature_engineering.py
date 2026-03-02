import pandas as pd
import numpy as np


def feature_engineering(X):
    """
    Feature engineering for the Consumer Complaints escalation prediction task.

    Transforms the raw complaints DataFrame by:
      1. Extracting temporal features from 'Date received'
      2. Computing response time (days from receipt to company assignment)
      3. Creating a binary flag AND character-length for the complaint narrative
      4. Extracting the broad geographic region from the ZIP code
      5. Creating binary flags from the 'Tags' column (Older American, Servicemember)
      6. Encoding 'Timely response?' as a binary numeric feature
      7. Dropping identifier, raw date, free-text, and target columns

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature DataFrame (must NOT contain the target column
        'Consumer disputed?' — that should be separated before calling).

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame containing only the engineered and
        retained columns, ready for imputation, scaling, and encoding.
    """
    X = X.copy()

    # ------------------------------------------------------------------
    # 1. Temporal features extracted from 'Date received'
    # ------------------------------------------------------------------
    if 'Date received' in X.columns:
        date_received = pd.to_datetime(X['Date received'], errors='coerce')
        X['received_month']   = date_received.dt.month.astype(float)
        X['received_year']    = date_received.dt.year.astype(float)
        X['received_weekday'] = date_received.dt.dayofweek.astype(float)  # 0=Mon, 6=Sun
        X['is_weekend']       = (date_received.dt.dayofweek >= 5).astype(float)
    else:
        X['received_month']   = np.nan
        X['received_year']    = np.nan
        X['received_weekday'] = np.nan
        X['is_weekend']       = np.nan

    # ------------------------------------------------------------------
    # 2. Response time: calendar days from receipt to company assignment
    #    Negative values (data entry errors) are clipped to 0.
    # ------------------------------------------------------------------
    if 'Date sent to company' in X.columns and 'Date received' in X.columns:
        date_sent = pd.to_datetime(X['Date sent to company'], errors='coerce')
        date_recv = pd.to_datetime(X['Date received'],        errors='coerce')
        X['response_time_days']     = (date_sent - date_recv).dt.days.clip(lower=0).astype(float)
        X['response_time_log']  = np.log1p(X['response_time_days'])
    else:
        X['response_time_days'] = np.nan
        X['response_time_log']  = np.nan

    # Drop raw date columns — information captured in engineered features
    X = X.drop(columns=['Date received', 'Date sent to company'], errors='ignore')

    # ------------------------------------------------------------------
    # 3. Narrative features: binary flag, character length, and log-length
    #    A longer narrative signals a more motivated, detail-oriented consumer
    #    who is more likely to escalate — richer than a simple presence flag.
    #    Log-length compresses the heavy right-skew (some narratives are very
    #    long) so the model can form better split thresholds.
    # ------------------------------------------------------------------
    if 'Consumer complaint narrative' in X.columns:
        X['has_narrative']        = X['Consumer complaint narrative'].notna().astype(int)
        X['narrative_length']     = (
            X['Consumer complaint narrative'].fillna('').str.len().astype(float)
        )
        X['narrative_length_log'] = np.log1p(X['narrative_length'])
        X = X.drop(columns=['Consumer complaint narrative'])
    else:
        X['has_narrative']        = 0
        X['narrative_length']     = 0.0
        X['narrative_length_log'] = 0.0

    # ------------------------------------------------------------------
    # 4. Geographic region: first digit of ZIP code (0–9)
    #    US ZIP codes are geographically ordered by first digit, giving
    #    9 broad regions (Northeast → West). Retaining this collapses
    #    ~30 000 unique ZIPs into a single numeric feature with real signal.
    # ------------------------------------------------------------------
    if 'ZIP code' in X.columns:
        zip_first = X['ZIP code'].astype(str).str.strip().str[0]
        X['zip_region'] = pd.to_numeric(zip_first, errors='coerce')
        X = X.drop(columns=['ZIP code'])
    else:
        X['zip_region'] = np.nan

    # ------------------------------------------------------------------
    # 5. Tags: binary indicator columns
    #    Original column has free text, e.g. "Older American, Servicemember"
    # ------------------------------------------------------------------
    if 'Tags' in X.columns:
        tags_str = X['Tags'].fillna('')
        X['is_older_american'] = (
            tags_str.str.contains('Older American', case=False, na=False).astype(int)
        )
        X['is_servicemember'] = (
            tags_str.str.contains('Servicemember', case=False, na=False).astype(int)
        )
        X = X.drop(columns=['Tags'])
    else:
        X['is_older_american'] = 0
        X['is_servicemember']  = 0

    # ------------------------------------------------------------------
    # 6. Company response quality: explicit binary flags
    #    'Company response to consumer' is already retained as a categorical
    #    feature (OHE). These dense binary flags reinforce the most important
    #    response categories as direct numeric signals for the model.
    #    — Q1.2 Insight 1 identified this as the strongest escalation driver.
    # ------------------------------------------------------------------
    if 'Company response to consumer' in X.columns:
        resp = X['Company response to consumer'].fillna('').str.lower()
        X['got_monetary_relief'] = resp.str.contains(
            'monetary relief', na=False
        ).astype(int)
        X['got_any_relief'] = (
            resp.str.contains('relief', na=False) &
            ~resp.str.contains('without relief', na=False)
        ).astype(int)
    else:
        X['got_monetary_relief'] = 0
        X['got_any_relief']      = 0

    # ------------------------------------------------------------------
    # 7. Timely response: Yes → 1, anything else → 0
    # ------------------------------------------------------------------
    if 'Timely response?' in X.columns:
        X['timely_response'] = (X['Timely response?'] == 'Yes').astype(int)
        X = X.drop(columns=['Timely response?'])
    else:
        X['timely_response'] = 0

    # ------------------------------------------------------------------
    # 8. Drop identifier and non-predictive columns
    #    'Consumer disputed?' is the target; it must never be a feature.
    # ------------------------------------------------------------------
    cols_to_drop = [
        'Complaint ID',        # row identifier — no predictive signal
        'Consumer disputed?',  # target variable — safety guard if included in X
    ]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    return X
