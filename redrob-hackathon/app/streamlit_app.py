import streamlit as st
import pandas as pd
import subprocess
import os
import yaml

st.set_page_config(page_title="Recommender Sandbox", layout="wide")

st.title("🚀 Enterprise AI Candidate Recommender")
st.markdown("This sandbox runs the end-to-end multi-stage ranking pipeline (FAISS + Cross-Encoder + Behavioral Math) on a small sample of candidates.")

st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader("Upload Candidates JSONL (Optional, defaults to pre-loaded)", type=["jsonl"])

if st.sidebar.button("Run End-to-End Ranking Pipeline"):
    st.info("Starting Pipeline...")
    
    # 1. Setup paths
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 2. Revert to original config to ensure we read job_description.docx
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config['paths']['jd_path'] = "data/raw/job_description.docx"
    with open("configs/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    # 3. Save uploaded JSONL or use default
    if uploaded_file is not None:
        with open("data/raw/helper_candidates.jsonl", "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # 4. Run Indexer
    with st.spinner("Step 1/2: Running Offline Indexer (Building FAISS & BM25)..."):
        result_indexer = subprocess.run(["python", "-m", "src.indexing.offline_indexer"], capture_output=True, text=True)
        if result_indexer.returncode != 0:
            st.error("Indexer Failed!")
            st.code(result_indexer.stderr)
            st.stop()
            
    # 5. Run Ranker
    with st.spinner("Step 2/2: Running Online Ranking Engine (Cross-Encoder & Behavioral Math)..."):
        result_ranker = subprocess.run(["python", "-m", "src.ranking.rank"], capture_output=True, text=True)
        if result_ranker.returncode != 0:
            st.error("Ranking Failed!")
            st.code(result_ranker.stderr)
            st.stop()
            
    st.success("✅ Ranking Complete!")
    
    # 6. Display Results
    output_path = "data/output/submission.csv"
        
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        st.subheader("🏆 Top Ranked Candidates")
        st.dataframe(df, use_container_width=True)
        
        with open(output_path, "rb") as f:
            st.download_button("Download submission.csv", f, file_name="submission.csv")
    else:
        st.error(f"Output CSV not found at {output_path}!")
