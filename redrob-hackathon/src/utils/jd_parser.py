import re

# Default canonical skill/technology vocabulary used to detect a JD's mandatory
# skills when no config-driven vocabulary is supplied. Covers the ML/AI/retrieval
# ecosystem broadly (not hand-tuned to one specific JD) so the same detector works
# if the JD text is swapped. Synonyms map a canonical token to alternate phrasings
# that should also count as a match.
# Split to mirror how JDs commonly self-tier skills ("things you absolutely need" vs
# "things we'd like but won't reject you for"). Lumping both into one bucket dilutes
# the mandatory-skill signal: a candidate with deep core retrieval/ranking substance
# scores nearly the same as one who only name-drops a trendy LLM-tooling term.
DEFAULT_SKILL_VOCABULARY = [
    "python", "pytorch", "tensorflow", "scikit-learn",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "pgvector",
    "sentence transformers", "embeddings", "semantic search", "vector search", "information retrieval",
    "hybrid search", "bm25", "learning to rank", "recommendation systems", "ranking",
]
DEFAULT_NICE_TO_HAVE_VOCABULARY = [
    "nlp", "llms", "rag", "langchain", "llamaindex", "haystack", "prompt engineering",
    "fine-tuning llms", "lora", "qlora", "peft",
]
DEFAULT_SKILL_SYNONYMS = {
    "sentence transformers": ["sentence-transformers", "sbert"],
    "pytorch": ["torch"],
    "vector search": ["vector database", "vector databases"],
    "hybrid search": ["hybrid retrieval"],
    "learning to rank": ["learning-to-rank", "ltr"],
    "fine-tuning llms": ["fine-tuning", "llm fine-tuning"],
}


def _matches(token: str, synonyms: dict, jd_lower: str) -> bool:
    variants = [token] + synonyms.get(token, [])
    return any(v in jd_lower for v in variants)


def parse_jd_rules(jd_text: str, config: dict | None = None) -> dict:
    """
    Parses the Job Description text to dynamically extract hard constraints.
    In a production environment, this function would call an LLM API (like OpenAI/Gemini)
    to extract JSON rules. For this offline Hackathon, we use keyword heuristics
    to extract the rules from the provided text.

    `config` (optional) is the `jd_parsing` section of config.yaml, allowing the
    skill vocabulary to be tuned per-JD without code changes. Falls back to a
    built-in generic vocabulary when not supplied.
    """
    jd_lower = jd_text.lower()
    config = config or {}
    skill_vocabulary = config.get("skill_vocabulary", DEFAULT_SKILL_VOCABULARY)
    nice_to_have_vocabulary = config.get("nice_to_have_vocabulary", DEFAULT_NICE_TO_HAVE_VOCABULARY)
    skill_synonyms = config.get("skill_synonyms", DEFAULT_SKILL_SYNONYMS)

    # Initialize the rules dictionary
    rules = {
        "banned_companies": set(),
        "mandatory_skills": set(),
        "nice_to_have_skills": set(),
        "min_job_duration_months": 0,
        "experience_band": None,
    }

    # 1. Detect Banned Consulting Firms
    # The JD mentions "People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)"
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    for firm in consulting_firms:
        if firm in jd_lower:
            rules["banned_companies"].add(firm)

    # 2. Detect Mandatory Skills — generic config-driven vocabulary scan (replaces the
    # old hardcoded "python" + fixed vector-db list so a swapped JD still works).
    for token in skill_vocabulary:
        if _matches(token, skill_synonyms, jd_lower):
            rules["mandatory_skills"].add(token)

    # 2b. Nice-to-have skills — matched separately so they can carry a smaller bonus
    # weight than true must-haves (see scorer.py's proportional bonus signal).
    for token in nice_to_have_vocabulary:
        if _matches(token, skill_synonyms, jd_lower):
            rules["nice_to_have_skills"].add(token)

    # 3. Detect Title-Chaser limits
    # "switching companies every 1.5 years" (18 months)
    if "1.5 years" in jd_lower or "18 months" in jd_lower:
        rules["min_job_duration_months"] = 18

    # 4. Detect a years-of-experience band, e.g. "5-9 years" / "5 to 9 years" / "minimum 5 years"
    band_match = re.search(r"(\d+)\s*[-–to]+\s*(\d+)\s*years", jd_lower)
    if band_match:
        rules["experience_band"] = (float(band_match.group(1)), float(band_match.group(2)))
    else:
        min_match = re.search(r"(?:minimum|at least)\s*(\d+)\s*years", jd_lower)
        if min_match:
            rules["experience_band"] = (float(min_match.group(1)), None)

    return rules
