import re

def parse_jd_rules(jd_text: str) -> dict:
    """
    Parses the Job Description text to dynamically extract hard constraints.
    In a production environment, this function would call an LLM API (like OpenAI/Gemini)
    to extract JSON rules. For this offline Hackathon, we use keyword heuristics 
    to extract the rules from the provided text.
    """
    jd_lower = jd_text.lower()
    
    # Initialize the rules dictionary
    rules = {
        "banned_companies": set(),
        "mandatory_skills": set(),
        "min_job_duration_months": 0
    }
    
    # 1. Detect Banned Consulting Firms
    # The JD mentions "People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)"
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    for firm in consulting_firms:
        if firm in jd_lower:
            rules["banned_companies"].add(firm)
            
    # 2. Detect Mandatory Skills
    vector_dbs = ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss"]
    for db in vector_dbs:
        if db in jd_lower:
            rules["mandatory_skills"].add(db)
            
    if "python" in jd_lower:
        rules["mandatory_skills"].add("python")
        
    # 3. Detect Title-Chaser limits
    # "switching companies every 1.5 years" (18 months)
    if "1.5 years" in jd_lower or "18 months" in jd_lower:
        rules["min_job_duration_months"] = 18
        
    return rules
