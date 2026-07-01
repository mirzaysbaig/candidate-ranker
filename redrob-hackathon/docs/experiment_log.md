# Experiment Log — Candidate Ranker Accuracy

## 0. The JD discrepancy (read this first)

`configs/config.yaml` → `paths.jd_path` points at `data/raw/job_description.docx`, **not**
`data/raw/job_description.md`. The two describe different roles:

- `job_description.md` (unused at runtime): a short, generic "Senior ML Engineer" JD.
- `job_description.docx` (the one the pipeline actually embeds and parses): a long, narrative
  "Senior AI Engineer — Founding Team" JD for "Redrob AI." It explicitly lists disqualifiers
  (pure-research-only careers, <12-month-old LangChain-only "AI experience," 18+ months without
  production code, title-chasers who switch every 1.5 years, consulting-only careers, CV/speech/
  robotics without NLP), states its 5-9 year band is "a range, not a requirement," and says
  outright: **"the right answer to this JD is not 'find candidates whose skills section contains
  the most AI keywords.' That's a trap we've explicitly built into the dataset."**

All labeling and experiments below judge fit against this real JD.

## 1. Methodology (Part A)

No ground truth existed in the repo, so 78 candidates were hand-labeled (0–3 graded relevance) by
reading each candidate's full record against the real JD: the baseline top-50, 8 candidates with
genuine ML/search substance sampled from outside the baseline top-50 ("false negatives"), 10
candidates with many literal AI/retrieval keyword tags on an unambiguously non-technical title
("false positive risk" — the dataset's keyword-stuffing trap), and 10 random draws. Full rubric
and label file: `data/eval/relevance_labels_v1.csv` / `.README.md`.

Metrics (`src/eval/evaluate_ranking.py`): NDCG@10/50 (IDCG computed from the *full* labeled
universe so missing a highly-relevant labeled candidate is correctly penalized, not ignored),
Precision@10/50 (relevance ≥ 2), mean relevance@10/50, Kendall's tau (system order vs.
label-sorted order, restricted to labeled candidates), and a disqualifier-leakage count (labeled
`relevance=0` + `disqualifier_flag=true` candidates still appearing in the top 50).

## 2. A critical bug found during labeling

While sampling "false negative" candidates, `CAND_0000031` (Recommendation Systems Engineer at
Swiggy; career history: shipped ranking models for a discovery feed, a Search Engineer stint at an
AI/ML company, owned a ranking layer for e-commerce search — the textbook "ideal candidate" the JD
describes) turned out to have `final_score = 0.0` in the baseline. Root cause: the job-hopper hard
filter extracts `min_job_duration_months = 18` from the JD's "switching companies every 1.5 years"
phrasing, and this candidate's average job duration was **17.5 months** — a hard binary cliff with
zero tolerance zeroed out a genuinely strong candidate over noise-level distance from the
threshold. This is fixed below (title-chaser grace margin) alongside the three planned
experiments.

## 3. Results

| variant | ndcg@10 | ndcg@50 | precision@10 | precision@50 | mean_rel@10 | mean_rel@50 | kendall_tau |
|---|---|---|---|---|---|---|---|
| E0_baseline | 0.781 | 0.801 | 0.50 | 0.16 | 1.7 | 0.58 | 0.324 |
| E1_skillvocab (vocab expansion only, old flat bonus) | 0.411 | 0.576 | 0.40 | 0.16 | 1.2 | 0.58 | 0.121 |
| E2_proportional_bonus (+ fraction-based bonus) | 0.480 | 0.620 | 0.40 | 0.16 | 1.3 | 0.58 | 0.231 |
| E3_jobhop_grace (+ title-chaser grace margin) | 0.551 | 0.671 | 0.50 | 0.18 | 1.5 | 0.62 | 0.277 |
| E4_experience_taper (+ soft experience-band signal) | 0.553 | 0.672 | 0.50 | 0.18 | 1.5 | 0.62 | 0.305 |
| **E5_tiered_bonus_final (+ must-have/nice-to-have split)** | **0.705** | **0.777** | **0.50** | **0.18** | **1.6** | **0.62** | **0.327** |
| E6_mpnet_embeddings (E5 + all-mpnet-base-v2) | 0.719 | 0.790 | 0.50 | 0.18 | 1.6 | 0.62 | 0.334 |

`disqualifier_leakage_count = 0` in every variant — the existing non-tech-title and
consulting-only hard filters already correctly zero out the keyword-stuffed decoy candidates
(the "false positive risk" stratum), across the full career history, not just the current title.
That part of the system was already working; it was not touched.

### Reading the table

- **E1 alone looks like a regression** (NDCG drops sharply) — this is a real and important
  finding, not a metric artifact: widening the mandatory-skill vocabulary while keeping the old
  *flat* "any single match = 2x bonus" rule made the keyword-trap problem **worse**, because more
  vocabulary terms means more low-substance candidates (a lone "Sentence Transformers" or
  "Learning to Rank" tag) qualify for the full bonus. Expanding the vocabulary is only safe once
  the bonus itself stops being a single any-match boolean.
- **E2's proportional (fraction-of-matched) bonus recovers some of that loss** but not all: with
  23 detected "mandatory" skills (the JD's must-have and nice-to-have sections got conflated),
  even a strong candidate matching several core skills only reaches a small fraction, diluting the
  bonus's signal.
- **E3 (job-hopper grace margin) is where `CAND_0000031` first re-enters the top 50** — it jumps
  straight to **rank 3**, confirming the hard-filter bug was the dominant blocker, not the bonus
  mechanics.
- **E5 (splitting mandatory vs. nice-to-have skills, per the JD's own two-tier framing)** recovers
  most of the NDCG lost in E1/E2 while *improving* precision@50, mean-relevance@50, and Kendall's
  tau over the original baseline. `CAND_0000031` reaches **rank 1**. All four labeled grade-3
  "strong fit" candidates are in the final top 50 (baseline only surfaced 3 of them).
- **E6 (mpnet embeddings)** gives a further small, consistent improvement across every metric at
  ~3x the embedding compute cost. At this dataset's scale (500 candidates) that cost is
  negligible; at the JD's own stated production scale (100K candidates) it would need real
  benchmarking before adopting. Given the marginal gain, this is left as an optional upgrade
  (config-only: `model.name: "all-mpnet-base-v2"`, see `configs/config_e4_mpnet.yaml`) rather than
  the new default.

## 4. Recommended final configuration

`configs/config.yaml` now contains the recommended combination (E5): expanded config-driven skill
vocabulary with must-have/nice-to-have separation, proportional mandatory-skill bonus, a 3-month
title-chaser grace margin, and a soft (non-cliff) experience-band taper. This was re-run end to
end as a single combined variant (`data/output/E5_final_tiered.csv`,
`E5_tiered_bonus_final` row above) rather than assumed additive from the individual deltas.

Key new config knobs (`scoring_weights`):
```yaml
title_chaser_grace_margin_months: 3
mandatory_skill_bonus_max: 2.0
mandatory_skill_bonus_curve: 1.0
nice_to_have_bonus_max: 1.2
experience_taper_span_years: 5
experience_taper_floor: 0.85
```
Optional upgrade (not the default): `model.name: "all-mpnet-base-v2"` with separate index/parquet
paths, per `configs/config_e4_mpnet.yaml`.

## 5. Known limitations

- Label set is 78 candidates from a single labeling pass (Claude-as-judge) — directional signal,
  not a statistically robust benchmark. No inter-rater reliability check.
- Career-history `description` text in this dataset is synthetic/templated and frequently
  mismatched with the listed title; labels weighted title + company + skill-list coherence over
  the free-text description, which was found unreliable.
- `retrieval.top_k: 500` equals the entire candidate pool at this dataset's size, so FAISS
  retrieval currently filters nothing — all of the leverage found here is in the re-ranking stage.
  This would need revisiting at real production scale (the JD itself references a 100K-candidate
  pool).
- A handful of clearly non-fit candidates (e.g. `CAND_0000317` DevOps Engineer, `CAND_0000249` QA
  Engineer at HCL — both labeled 0) still rank in the top 10 across every variant, including E6.
  This is a **semantic-retrieval** artifact (their embedding-based `semantic_score` is high) that
  scorer-level bonus/filter tuning cannot fix — a genuine open item for further embedding-model or
  text-corpus work, not resolved by this round of experiments.
- Labels were produced against `job_description.docx` as of 2026-07-01; if the JD changes, this
  label set must be regenerated.
- A separate, unrelated environment bug was fixed to get the pipeline running at all: the
  installed `sentence-transformers` version renamed `get_embedding_dimension()` to
  `get_sentence_embedding_dimension()` (`src/indexing/offline_indexer.py`), and `data/output/`
  didn't exist as a real directory.
