# RedRob Hackathon Project

Welcome to the RedRob Hackathon submission template. This repository contains the framework for indexing, retrieving, and ranking candidates based on a job description.

## Directory Structure

```
redrob-hackathon/
├── .venv/                     # Python virtual environment (ignored by git)
├── .gitignore                 # Git ignore file
├── requirements.txt           # Project Python dependencies
├── README.md                  # Project overview and guide
├── submission_metadata.yaml   # Required submission metadata for organizers
│
├── data/
│   ├── raw/                   # Raw input data (e.g. candidates.jsonl, job_description.md)
│   ├── processed/             # Processed datasets (e.g. parquet files, FAISS vector indexes)
│   └── output/                # Finished ranking CSV files (e.g. team_xxx.csv)
│
├── app/
│   └── streamlit_app.py       # Streamlit dashboard for visual verification
│
├── src/                       # Main source code package
│   ├── __init__.py            # Root init to expose package
│   ├── indexing/              # Processing and vector indexing scripts
│   │   ├── __init__.py
│   │   └── offline_indexer.py
│   ├── ranking/               # Candidate evaluation and ranking
│   │   ├── __init__.py
│   │   └── rank.py            # Primary 5-minute ranking executable
│   ├── embeddings/            # Vector embeddings generation
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   ├── retrieval/             # FAISS indexing and vector search
│   │   ├── __init__.py
│   │   └── faiss_search.py
│   ├── models/                # Scorer, calibration, and ranking formulas
│   │   ├── __init__.py
│   │   └── scorer.py
│   └── utils/                 # Prompts, logging, and helpers
│       ├── __init__.py
│       └── helpers.py
│
├── notebooks/                 # Jupyter notebooks for experiments
│   └── experiments.ipynb
│
└── configs/                   # System configurations
    └── config.yaml            # FAISS configuration, embedding dimensions, thresholds
```

## Setup Instructions

1. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate Virtual Environment**:
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - On Linux/macOS:
     ```bash
     source .venv/Scripts/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Streamlit Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
