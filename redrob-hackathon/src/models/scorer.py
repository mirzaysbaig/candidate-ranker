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

def apply_behavioral_math(
    top_candidates_df: pd.DataFrame,
    jd_rules: dict,
    evaluation_date_str: str = '2026-06-25',
    config_path: str = 'configs/config.yaml'
) -> pd.DataFrame:
    """
    Applies continuous mathematical curves strictly using variables from config.yaml.
    
    Computes 14 behavioral multipliers and combines them multiplicatively:
      final_score = semantic_score × Π(all multipliers)
    
    Each multiplier outputs a value around 1.0:
      > 1.0 = boost the candidate
      = 1.0 = neutral (no effect)
      < 1.0 = penalize the candidate
    
    Signal Map:
    ───────────
     1. activity_mult       — Exponential time decay on last_active_date
     2. notice_mult         — Inverted sigmoid on notice_period_days
     3. response_mult       — Linear scaling on recruiter_response_rate
     4. github_mult         — Logarithmic scaling on github_activity_score
     5. open_to_work_bonus  — Flat boolean bonus on open_to_work_flag
     6. assessment_mult     — Sigmoid normalization on avg_assessment_score
     7. completeness_mult   — Power curve on profile_completeness_score
     8. education_mult      — Discrete tier bonus on education_tier
     9. cert_mult           — Logarithmic bonus on certifications_count
    10. interview_mult      — Linear reliability on interview_completion_rate
    11. social_mult         — Log social proof on endorsements + connections
    12. trust_mult          — Flat compound bonus on verified flags
    13. offer_mult          — Sigmoid reliability on offer_acceptance_rate
    14. demand_mult         — Log market signal on search_appearance + saves
    """
    
    scored_df = top_candidates_df.copy()
    eval_date = pd.to_datetime(evaluation_date_str)
    
    # Strictly load the configuration dictionary
    config = load_config(config_path)
    
    # If 'scoring_weights' doesn't exist in the YAML, this will intentionally throw a KeyError
    w = config['scoring_weights'] 
    
    # =================================================================
    # 1. The Availability Multiplier (Exponential Time Decay)
    # Formula: e^(-rate * days_inactive)
    # =================================================================
    last_active = pd.to_datetime(scored_df['last_active_date'], errors='coerce')
    days_inactive = (eval_date - last_active).dt.days.fillna(365)
    
    # STRICT READ: Pulls directly from YAML with NO hardcoded fallbacks
    decay_rate = w['time_decay_rate'] 
    scored_df['activity_mult'] = np.exp(-decay_rate * days_inactive)

    # =================================================================
    # 2. The Notice Period Modifier (Inverted Sigmoid)
    # Formula: 1 - (0.5 / (1 + e^(-steepness * (days - midpoint))))
    # =================================================================
    notice_days = scored_df['notice_period_days'].fillna(90) 
    
    # STRICT READ
    np_mid = w['notice_period_midpoint']
    np_steep = w['notice_period_steepness']
    
    scored_df['notice_mult'] = 1 - (0.5 / (1 + np.exp(-np_steep * (notice_days - np_mid))))

    # =================================================================
    # 3. The Engagement Modifier (Linear Scaling)
    # Formula: base + (scale * response_rate)
    # =================================================================
    response_rate = scored_df['recruiter_response_rate'].fillna(0)
    
    # STRICT READ
    resp_base = w['response_rate_base']
    resp_scale = w['response_rate_scale']
    
    scored_df['response_mult'] = resp_base + (resp_scale * response_rate)

    # =================================================================
    # 4. The Intent Modifier (Logarithmic GitHub Scaling)
    # Formula: 1.0 + (mult * log(1 + score))
    # =================================================================
    safe_github = np.maximum(0, scored_df['github_activity_score'].fillna(0))
    
    # STRICT READ
    gh_mult = w['github_log_multiplier']
    scored_df['github_mult'] = 1.0 + (gh_mult * np.log1p(safe_github))

    # =================================================================
    # 5. The Open-to-Work Bonus (Flat Boolean)
    # =================================================================
    # STRICT READ
    otw_bonus = w['open_to_work_bonus']
    scored_df['open_to_work_bonus'] = np.where(scored_df['open_to_work_flag'] == True, otw_bonus, 1.0)

    # =================================================================
    # 6. Skill Assessment Competency (Sigmoid Normalization)
    # Formula: base + (range / (1 + e^(-steepness * (score - midpoint))))
    # Industry: HackerRank/Codility-style percentile scoring
    # =================================================================
    raw_assessment = scored_df['avg_assessment_score'].fillna(-1)
    
    # STRICT READ
    assess_mid = w['assessment_midpoint']
    assess_steep = w['assessment_steepness']
    assess_base = w['assessment_base']
    assess_range = w['assessment_range']
    
    sigmoid_score = assess_base + (assess_range / (1 + np.exp(-assess_steep * (raw_assessment - assess_mid))))
    # Candidates with no assessment (-1) get neutral 1.0
    scored_df['assessment_mult'] = np.where(raw_assessment < 0, 1.0, sigmoid_score)

    # =================================================================
    # 7. Profile Completeness (Power Scaling)
    # Formula: (score / 100) ^ exponent
    # Industry: LinkedIn/Naukri profile strength meters
    # =================================================================
    completeness = scored_df['profile_completeness_score'].fillna(0).clip(1, 100)
    
    # STRICT READ
    comp_exp = w['completeness_exponent']
    
    scored_df['completeness_mult'] = np.power(completeness / 100.0, comp_exp)

    # =================================================================
    # 8. Education Tier (Discrete Bonus Map)
    # Industry: Indian recruitment tier system (IIT/NIT/State)
    # =================================================================
    # STRICT READ
    tier_bonus_map = w['education_tier_bonus']
    
    scored_df['education_mult'] = scored_df['education_tier'].map(tier_bonus_map).fillna(1.0)

    # =================================================================
    # 9. Certifications (Logarithmic Bonus)
    # Formula: 1.0 + (mult * log(1 + count))
    # Industry: Diminishing returns on credential stacking
    # =================================================================
    cert_count = scored_df['certifications_count'].fillna(0).clip(lower=0)
    
    # STRICT READ
    cert_mult_val = w['cert_log_multiplier']
    
    scored_df['cert_mult'] = 1.0 + (cert_mult_val * np.log1p(cert_count))

    # =================================================================
    # 10. Interview Completion Rate (Linear Reliability)
    # Formula: base + (scale * completion_rate)
    # Industry: No-show tracking in ATS systems
    # =================================================================
    interview_rate = scored_df['interview_completion_rate'].fillna(0.5)
    
    # STRICT READ
    intv_base = w['interview_completion_base']
    intv_scale = w['interview_completion_scale']
    
    scored_df['interview_mult'] = intv_base + (intv_scale * interview_rate)

    # =================================================================
    # 11. Social Proof (Logarithmic — Endorsements + Connections)
    # Formula: 1.0 + (e_mult * log(1+endorse)) + (c_mult * log(1+connect))
    # Industry: LinkedIn algorithm log-normalization of social counts
    # =================================================================
    endorsements = scored_df['endorsements_received'].fillna(0).clip(lower=0)
    connections = scored_df['connection_count'].fillna(0).clip(lower=0)
    
    # STRICT READ
    endorse_mult = w['endorsement_log_multiplier']
    connect_mult = w['connection_log_multiplier']
    
    scored_df['social_mult'] = (
        1.0
        + (endorse_mult * np.log1p(endorsements))
        + (connect_mult * np.log1p(connections))
    )

    # =================================================================
    # 12. Verification Trust Score (Flat Compound Bonus)
    # Industry: Platform verification reduces fraud risk in ATS
    # =================================================================
    # STRICT READ
    email_bonus = w['verified_email_bonus']
    phone_bonus = w['verified_phone_bonus']
    linkedin_bonus = w['linkedin_connected_bonus']
    
    trust_base = pd.Series(1.0, index=scored_df.index)
    trust_base += np.where(scored_df['verified_email'] == True, email_bonus, 0.0)
    trust_base += np.where(scored_df['verified_phone'] == True, phone_bonus, 0.0)
    trust_base += np.where(scored_df['linkedin_connected'] == True, linkedin_bonus, 0.0)
    scored_df['trust_mult'] = trust_base

    # =================================================================
    # 13. Offer Acceptance Rate (Sigmoid Reliability)
    # Formula: base + (range / (1 + e^(-steepness * (rate - midpoint))))
    # Industry: Pipeline efficiency — serial rejectors waste recruiter time
    # =================================================================
    raw_offer_rate = scored_df['offer_acceptance_rate'].fillna(-1)
    
    # STRICT READ
    offer_base = w['offer_acceptance_base']
    offer_range = w['offer_acceptance_range']
    offer_mid = w['offer_acceptance_midpoint']
    offer_steep = w['offer_acceptance_steepness']
    
    offer_sigmoid = offer_base + (offer_range / (1 + np.exp(-offer_steep * (raw_offer_rate - offer_mid))))
    # Candidates with no offer history (-1) get neutral 1.0
    scored_df['offer_mult'] = np.where(raw_offer_rate < 0, 1.0, offer_sigmoid)

    # =================================================================
    # 14. Market Demand Signal (Logarithmic — Search + Saves)
    # Formula: 1.0 + (s_mult * log(1+appearances)) + (r_mult * log(1+saves))
    # Industry: Recruiter demand as a market validation signal
    # =================================================================
    search_appearances = scored_df['search_appearance_30d'].fillna(0).clip(lower=0)
    saved_by_recruiters = scored_df['saved_by_recruiters_30d'].fillna(0).clip(lower=0)
    
    # STRICT READ
    search_mult = w['search_appearance_log_multiplier']
    saved_mult = w['saved_by_recruiters_log_multiplier']
    
    scored_df['demand_mult'] = (
        1.0
        + (search_mult * np.log1p(search_appearances))
        + (saved_mult * np.log1p(saved_by_recruiters))
    )

    # =================================================================
    # 15. JD DETERMINISTIC HARD FILTERS & BONUSES
    # =================================================================
    # Honeypot Filter
    honeypot_flags = scored_df.get('honeypot_flags', 0)
    scored_df['jd_hard_mult'] = np.where(honeypot_flags > 0, 0.0, 1.0)
    
    # Title-Chaser Filter
    # A grace margin avoids hard-zeroing candidates whose average job duration is only
    # marginally below the threshold (e.g. 17.5mo vs an 18mo cutoff extracted from JD
    # phrasing like "every 1.5 years") -- noise at that scale shouldn't be treated the
    # same as someone who visibly job-hops every year for title escalation.
    min_dur = jd_rules.get("min_job_duration_months", 0)
    if min_dur > 0:
        grace_margin = w.get('title_chaser_grace_margin_months', 0)
        avg_dur = scored_df.get('avg_job_duration_months', 999)
        scored_df['jd_hard_mult'] = np.where(avg_dur < (min_dur - grace_margin), 0.0, scored_df['jd_hard_mult'])

    # Consulting Firm Ban Filter (Industry-based)
    # The JD strictly says "only worked at consulting firms... if you have prior product-company experience, that's fine"
    def is_consulting_only(industries_str):
        if not isinstance(industries_str, str) or not industries_str:
            return False
        inds = [i.strip() for i in industries_str.split(',') if i.strip()]
        if not inds:
            return False
        # If EVERY single job they've ever had is 'it services'
        return all(i == 'it services' for i in inds)
        
    consulting_mask = scored_df['past_industries'].apply(is_consulting_only)
    scored_df['jd_hard_mult'] = np.where(consulting_mask, 0.0, scored_df['jd_hard_mult'])
    
    # Role Mismatch Penalty (Sales/HR/Non-Software candidates)
    # If the candidate has ZERO engineering/developer/data roles in their entire career
    def is_non_tech(titles_str):
        if not isinstance(titles_str, str) or not titles_str:
            return False
        tech_keywords = ['software', 'backend', 'frontend', 'developer', 'data', 'ml', 'machine learning', 'ai', 'architect', 'programmer', 'cloud', 'devops']
        return not any(tech in titles_str for tech in tech_keywords)
        
    non_tech_mask = scored_df['all_job_titles'].apply(is_non_tech)
    # Severely penalize pure non-tech careers (e.g. 100% Sales Managers)
    scored_df['jd_hard_mult'] = np.where(non_tech_mask, 0.0, scored_df['jd_hard_mult'])

    # Mandatory Skills Bonus (proportional, not a flat any-match 2x)
    # A candidate matching 1-of-6 mandatory skills previously got the identical bonus
    # as a candidate matching 6-of-6, which rewards a single lucky keyword hit as much
    # as genuine broad coverage. Bonus now scales with the fraction of mandatory
    # skills actually present, with a diminishing-returns curve exponent (consistent
    # with the file's other log-based diminishing-returns signals).
    mandatory_skills = jd_rules.get("mandatory_skills", set())
    if mandatory_skills:
        def frac_matched(skills_str):
            if not isinstance(skills_str, str) or not skills_str:
                return 0.0
            hits = sum(1 for req in mandatory_skills if req in skills_str)
            return hits / len(mandatory_skills)

        match_fraction = scored_df['all_skills'].apply(frac_matched)
        bonus_max = w.get('mandatory_skill_bonus_max', 2.0)
        bonus_curve = w.get('mandatory_skill_bonus_curve', 1.0)
        scored_df['jd_bonus_mult'] = 1.0 + (bonus_max - 1.0) * (match_fraction ** bonus_curve)
    else:
        scored_df['jd_bonus_mult'] = 1.0

    # Nice-to-have skills bonus: smaller magnitude, kept separate from the mandatory
    # bonus so LLM-tooling name-drops (LangChain, LoRA, ...) can't buy the same credit
    # as genuine core retrieval/ranking substance.
    nice_to_have_skills = jd_rules.get("nice_to_have_skills", set())
    if nice_to_have_skills:
        def frac_nice(skills_str):
            if not isinstance(skills_str, str) or not skills_str:
                return 0.0
            hits = sum(1 for req in nice_to_have_skills if req in skills_str)
            return hits / len(nice_to_have_skills)

        nice_fraction = scored_df['all_skills'].apply(frac_nice)
        nice_bonus_max = w.get('nice_to_have_bonus_max', 1.2)
        scored_df['jd_bonus_mult'] = scored_df['jd_bonus_mult'] * (1.0 + (nice_bonus_max - 1.0) * nice_fraction)

    # =================================================================
    # 16. Experience-Fit Multiplier (soft curve, not a hard cliff)
    # The JD's stated years-of-experience band is explicitly "a range, not a
    # requirement" -- so this must taper gently rather than zero out candidates
    # outside it. Full credit (1.0) inside the band, linear taper down to a floor
    # a configurable number of years outside either edge.
    # =================================================================
    experience_band = jd_rules.get("experience_band")
    if experience_band:
        min_years, max_years = experience_band
        years_exp = scored_df.get('years_of_experience', 0).fillna(0)
        taper_span = w.get('experience_taper_span_years', 5)
        floor = w.get('experience_taper_floor', 0.85)

        def taper(y):
            if y < min_years:
                shortfall = min_years - y
            elif max_years is not None and y > max_years:
                shortfall = y - max_years
            else:
                return 1.0
            frac = min(shortfall / taper_span, 1.0)
            return 1.0 - frac * (1.0 - floor)

        scored_df['experience_fit_mult'] = years_exp.apply(taper)
    else:
        scored_df['experience_fit_mult'] = 1.0

    # =================================================================
    # FINAL SCORE: Multiplicative composition of all signals
    # =================================================================
    scored_df['final_score'] = (
        scored_df['semantic_score']
        * scored_df['activity_mult']
        * scored_df['notice_mult']
        * scored_df['response_mult']
        * scored_df['github_mult']
        * scored_df['open_to_work_bonus']
        * scored_df['assessment_mult']
        * scored_df['completeness_mult']
        * scored_df['education_mult']
        * scored_df['cert_mult']
        * scored_df['interview_mult']
        * scored_df['social_mult']
        * scored_df['offer_mult']
        * scored_df['demand_mult']
        * scored_df['jd_hard_mult']
        * scored_df['jd_bonus_mult']
        * scored_df['experience_fit_mult']
    )
    
    return scored_df