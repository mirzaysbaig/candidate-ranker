"""
Dynamic JD Parser
═════════════════
Extracts structured rules from ANY job description text at runtime.
All classification knowledge comes from configs/filters.yaml — 
no hardcoded company names, skill lists, or keyword matches.

Usage:
    from src.utils.jd_parser import parse_jd_rules
    rules = parse_jd_rules(jd_text)
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, Set, Tuple


def load_filters(filters_path: str = 'configs/filters.yaml') -> dict:
    """Loads the dynamic classification filters from YAML."""
    with open(filters_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_jd_rules(
    jd_text: str,
    filters_path: str = 'configs/filters.yaml'
) -> dict:
    """
    Dynamically extracts structured constraints from any job description.
    
    All classification rules (skill domains, industry types, role categories)
    are read from filters.yaml at runtime — nothing is hardcoded.
    
    Returns:
        dict with keys:
            penalized_industries (set):     Industry values to penalize
            required_skill_domains (set):   Skill domains the JD requires
            required_role_categories (set): Role types relevant to the JD
            mandatory_skills (set):         Specific skill names from JD
            nice_to_have_skills (set):      Skills the JD mentions as optional
            min_job_duration_months (int):  Minimum tenure per job
            experience_range (tuple):       (min_years, max_years) from JD
            career_substance_keywords (list): Keywords indicating real domain work
            jd_industry_context (set):      Product industries mentioned in JD
    """
    filters = load_filters(filters_path)
    jd_lower = jd_text.lower()
    
    rules: Dict[str, Any] = {
        "penalized_industries": set(),
        "required_skill_domains": set(),
        "required_role_categories": set(),
        "mandatory_skills": set(),
        "nice_to_have_skills": set(),
        "min_job_duration_months": 0,
        "experience_range": (0, 99),
        "career_substance_keywords": [],
        "jd_industry_context": set(),
    }
    

    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. DETECT INDUSTRY PENALTIES
    # If the JD mentions consulting/services negatively → penalize those industries
    # ═══════════════════════════════════════════════════════════════════════
    service_industries = filters.get('industry_classifications', {}).get('services', [])
    product_industries = filters.get('industry_classifications', {}).get('product_indicators', [])
    
    # Detect if JD explicitly discourages consulting backgrounds
    jd_patterns = filters.get('jd_pattern_matching', {})
    consulting_negative_patterns = jd_patterns.get('consulting_negative_patterns', [])
    
    for pattern in consulting_negative_patterns:
        if re.search(pattern, jd_lower):
            rules["penalized_industries"] = set(service_industries)
            break
    
    # Detect product company context in JD
    for industry in product_industries:
        if industry.lower() in jd_lower:
            rules["jd_industry_context"].add(industry.lower())
    

    
    # ═══════════════════════════════════════════════════════════════════════
    # 4. EXTRACT EXPERIENCE RANGE
    # ═══════════════════════════════════════════════════════════════════════
    exp_range_patterns = jd_patterns.get('experience_range_patterns', [])
    for pattern in exp_range_patterns:
        match = re.search(pattern, jd_lower)
        if match:
            rules["experience_range"] = (int(match.group(1)), int(match.group(2)))
            break
            
    if rules["experience_range"] == (0, 99):
        exp_min_patterns = jd_patterns.get('experience_min_patterns', [])
        for pattern in exp_min_patterns:
            match = re.search(pattern, jd_lower)
            if match:
                min_y = int(match.group(1))
                rules["experience_range"] = (min_y, min_y + 10)
                break
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. DETECT TITLE-CHASER / MINIMUM TENURE PATTERNS
    # ═══════════════════════════════════════════════════════════════════════
    title_chaser_patterns = jd_patterns.get('title_chaser_patterns', [])
    for pattern in title_chaser_patterns:
        match = re.search(pattern, jd_lower)
        if match:
            val = float(match.group(1))
            # if the pattern has 'years' in it in the yaml, we assume years, else months
            if 'year' in pattern:
                rules["min_job_duration_months"] = int(val * 12)
            else:
                rules["min_job_duration_months"] = int(val)
            break
    

    

    
    return rules
