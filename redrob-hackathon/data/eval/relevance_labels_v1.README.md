# Relevance labels v1 — labeling rubric

Static, one-time, offline ground-truth labels for evaluating candidate-ranker output. The pipeline
never calls an LLM at runtime; these labels were produced by a human/Claude reviewer reading each
candidate's full JSON record side-by-side with the real JD actually consumed by the pipeline
(`data/raw/job_description.docx`, a "Senior AI Engineer — Founding Team" role at "Redrob AI" —
**not** `data/raw/job_description.md`, which is unused at runtime and describes a different,
generic "Senior ML Engineer" role).

## Sample construction (78 candidates)

1. `baseline_top50` (50) — the full top-50 output of an unmodified baseline run (`E0`). Captures
   what today's system currently over/under-rates.
2. `false_negative_candidate` (8) — sampled from outside the baseline top 50 for having titles or
   career-history entries suggesting genuine ML/search/ranking substance (e.g. "Recommendation
   Systems Engineer", "ML Engineer" adjacent skill clusters at product companies), to test whether
   the system is missing real fits.
3. `false_positive_risk` (10) — sampled for having many literal AI/retrieval keyword skill tags
   (FAISS, Pinecone, RAG, LangChain, Recommendation Systems, etc.) attached to an unambiguously
   non-technical title/career (Graphic Designer, Customer Support, HR Manager, Business Analyst,
   etc.), to test whether the mandatory-skill bonus can be gamed by keyword-stuffing alone. The
   JD's own text explicitly names this pattern as a deliberate trap in the dataset.
4. `random` (10) — unbiased random draws from the remaining pool, needed to compute
   precision/recall meaningfully rather than only rank correlation among already-plausible
   candidates.

## Relevance scale (0–3, graded — enables NDCG)

- **3 — Strong fit.** Matches the JD's "ideal candidate" description: real production experience
  shipping a ranking/search/recommendation system at a product company (not pure consulting/IT
  services), roughly in or near the 5–9 year band, no disqualifiers triggered.
- **2 — Plausible/partial fit.** Real software/data engineering background with genuine (not just
  listed) exposure to search/ranking/ML infrastructure at a product company, or a real ML/AI title
  missing 1–2 must-have signals (e.g. below the experience band). No hard disqualifiers.
- **1 — Weak/marginal.** Tangentially technical; some genuine but thin ML-adjacent exposure, or a
  real engineering role at a consulting/IT-services company with ML skill tags but no supporting
  production narrative. Not a hard disqualifier, but not a real fit either.
- **0 — Not a fit.** Either an explicit disqualifier (non-technical title/career padded with
  AI/retrieval keyword tags — the dataset's built-in "keyword trap"; consulting-only career; no
  genuine technical background) or simply irrelevant (no meaningful ML/search signal at all).

`disqualifier_flag=true` marks labels where a specific JD-stated disqualifier was the deciding
factor (used as a hard-filter regression check, separate from the graded relevance score).

## Known limitations

- Single labeler (one Claude review pass), small N (78) — directional signal only, not a
  statistically robust benchmark.
- Career-history `description` text in this dataset is synthetic/templated and frequently
  mismatched with the listed title (e.g. a "Graphic Designer" entry with a sales-quota
  description) — labeling therefore weighted title + company + skill-list coherence over the
  free-text description, which was found to be unreliable signal.
- Labels reflect fit against the real `job_description.docx` content as of 2026-07-01. If the JD
  changes, this label set must be regenerated.
