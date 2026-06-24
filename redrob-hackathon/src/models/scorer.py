import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """
    Strictly loads the YAML configuration file. 
    If the file is missing, it will intentionally raise an error (Fail-Fast).
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def apply_behavioral_math(df: pd.DataFrame, evaluation_date_str: str = '2026-06-19', config_path: str = 'configs/config.yaml') -> pd.DataFrame:
    """
    Applies continuous mathematical curves strictly using variables from config.yaml.
    """
    
    scored_df = df.copy()
    eval_date = pd.to_datetime(evaluation_date_str)
    
    # Strictly load the configuration dictionary
    config = load_config(config_path)
    
    # If 'scoring_weights' doesn't exist in the YAML, this will intentionally throw a KeyError
    w = config['scoring_weights'] 
    
    # ---------------------------------------------------------
    # 1. The Availability Multiplier (Exponential Time Decay)
    # ---------------------------------------------------------
    last_active = pd.to_datetime(scored_df['last_active_date'], errors='coerce')
    days_inactive = (eval_date - last_active).dt.days.fillna(365)
    
    # STRICT READ: Pulls directly from YAML with NO hardcoded fallbacks
    decay_rate = w['time_decay_rate'] 
    scored_df['activity_mult'] = np.exp(-decay_rate * days_inactive)

    # ---------------------------------------------------------
    # 2. The Notice Period Modifier (Inverted Sigmoid)
    # ---------------------------------------------------------
    notice_days = scored_df['notice_period_days'].fillna(90) 
    
    # STRICT READ
    np_mid = w['notice_period_midpoint']
    np_steep = w['notice_period_steepness']
    
    scored_df['notice_mult'] = 1 - (0.5 / (1 + np.exp(-np_steep * (notice_days - np_mid))))

    # ---------------------------------------------------------
    # 3. The Engagement Modifier (Linear Scaling)
    # ---------------------------------------------------------
    response_rate = scored_df['recruiter_response_rate'].fillna(0)
    
    # STRICT READ (You will notice the 0.7 and 0.4 are completely gone from this file!)
    resp_base = w['response_rate_base']
    resp_scale = w['response_rate_scale']
    
    scored_df['response_mult'] = resp_base + (resp_scale * response_rate)

    # ---------------------------------------------------------
    # 4. The Intent Modifier (Logarithmic GitHub Scaling)
    # ---------------------------------------------------------
    safe_github = np.maximum(0, scored_df['github_activity_score'].fillna(0))
    
    # STRICT READ
    gh_mult = w['github_log_multiplier']
    scored_df['github_mult'] = 1.0 + (gh_mult * np.log1p(safe_github))

    # STRICT READ
    otw_bonus = w['open_to_work_bonus']
    scored_df['open_to_work_bonus'] = np.where(scored_df['open_to_work_flag'] == True, otw_bonus, 1.0)

    # ---------------------------------------------------------
    # 5. The Final Calculation
    # ---------------------------------------------------------
    scored_df['final_score'] = (
        scored_df['semantic_score'] * scored_df['activity_mult'] * scored_df['notice_mult'] * scored_df['response_mult'] * scored_df['github_mult'] * scored_df['open_to_work_bonus']
    )
    
    return scored_df