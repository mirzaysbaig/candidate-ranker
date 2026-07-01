# Experiments — Simple Summary

## What we were trying to do

Make the candidate-ranking system actually good at picking the right people for the job, and
prove it with numbers instead of guessing.

## The first problem: there was no way to check if the system was "accurate"

The project had no test set, no scoring, nothing that said "this ranking is good" or "this
ranking is bad." So before changing anything, we had to build a way to measure quality.

**What we did:** We read the real job description and manually graded 78 candidates from
0 to 3 (0 = not a fit, 3 = great fit), based on whether their actual career made sense for the
role — not just whether they had the right buzzwords on their resume. This became our answer key.
Then we wrote a script that compares any ranking output against this answer key and spits out
scores (how many good candidates made the top 10/50, etc.).

## The second problem: the real job description was hiding in a different file

The config was pointing to a Word document (`job_description.docx`), not the `.md` file everyone
would assume is the JD. The Word doc is a much more detailed, trickier posting for a "Senior AI
Engineer" role — and it literally says, in plain English: *"the right answer is not to find
candidates whose skills list has the most AI keywords. That's a trap we built into the data."*
So the dataset was deliberately designed to punish lazy keyword-matching.

## The big bug we found

While manually checking candidates, we found the single best-fit person in the entire dataset — a
"Recommendation Systems Engineer" who had literally shipped ranking systems before — was
getting a **score of zero**. Not just ranked low. Zero.

**Why:** the system has a rule that penalizes "job hoppers" (people who switch jobs too fast).
This person's average time per job was 17.5 months, and the rule's cutoff was 18 months. So a
half-month difference was treated exactly the same as someone who switches jobs every 3 months —
a real bug, not a design choice.

## The fixes we made

1. **Widened what counts as a "must-have skill."** Before, the system only recognized "python"
   and a short hardcoded list of database names. Now it recognizes the full range of AI/ML/search
   skills mentioned in the JD.

2. **Stopped treating "1 matching skill" the same as "6 matching skills."** Before, having just
   one relevant skill gave you the exact same 2x score boost as having all of them. Now the boost
   scales with how many you actually have.

3. **Separated "must-have" skills from "nice-to-have" skills**, because the JD itself makes that
   distinction — trendy tools like LangChain/LoRA are explicitly called out as "won't reject you
   for missing this," so they shouldn't count as much as core skills like Python or vector search.

4. **Gave the job-hopper rule some breathing room** (a small grace margin) instead of a hard
   cutoff, so people right at the border aren't unfairly zeroed out.

5. **Made the "years of experience" requirement softer**, since the JD itself says its 5-9 year
   range is "a range, not a requirement," not a hard rule.

6. **Tried a better AI model for matching resumes to the job description** (a small extra
   improvement, optional — costs more compute for a small gain).

## Did it work?

Yes. Some numbers from our test:

- Before: only 3 out of 4 genuinely great-fit candidates showed up in the top 50 at all.
- After: all 4 show up, and the single best candidate (the one who was scoring zero) is now
  ranked **#1**.
- Precision, average candidate quality, and overall ranking agreement with our manual grading all
  improved.
- Importantly: the "keyword-stuffed but obviously wrong" candidates (e.g. a Graphic Designer with
  20 AI buzzwords slapped on their profile) are still correctly filtered out — we didn't break
  that part while fixing the rest.

## What we tried and it made things *worse* at first (worth knowing)

Just widening the list of "must-have skills" without also fixing how the bonus is calculated
actually made rankings worse — because more recognized keywords meant more low-quality
candidates could grab the same flat bonus just by having one matching tag. This only got fixed
once we also made the bonus scale with how many skills actually matched (fix #2/#3 above).

## What's still not perfect (didn't fix, flagging honestly)

A few clearly wrong candidates (like a DevOps Engineer with no real AI background) still sneak
into the top 10 because the underlying AI text-matching model thinks their resume "sounds similar"
to the job description. That's a deeper issue with the matching model itself, not something we
could fix by adjusting scoring rules — would need a stronger/different embedding model and more
testing to fully solve.

## Where to look for the details

- `docs/experiment_log.md` — the full technical writeup with all metrics and file references.
- `data/eval/relevance_labels_v1.csv` — our manual grading of 78 candidates (the "answer key").
- `data/eval/experiment_results.csv` — the raw scores for every experiment we ran.
- `src/eval/evaluate_ranking.py` — the script that grades any ranking output against the answer key.
