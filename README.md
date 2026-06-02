# ResumeMax

Automated Resume ATS Score Optimization via NLP and Semantic Embeddings.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Layout

```
resumemax/
  src/resumemax/        # package source
  data/
    resumes/            # input resumes (pdf, docx, txt)
    job_descriptions/   # JD text files
    skills_taxonomy/    # ESCO / O*NET / Lightcast / SkillSpan
    lexicons/           # action verbs, buzzwords
    gold_standard/      # human-rated resume/JD pairs (see docs/GOLD_STANDARD.md)
  docs/
    AUDIT_RULES.md      # formal spec for every rule-based audit check
    LATENCY.md          # < 3 s end-to-end budget and optimizations
    GOLD_STANDARD.md    # annotation methodology + correlation protocol
    EXPLAINABILITY.md   # faithful explanations for every final score
  scripts/
    bench_latency.py    # per-stage latency benchmark
  tests/
  notebooks/
  outputs/
```

## CLI

```bash
# Score a resume against a JD
PYTHONPATH=src python -m resumemax.cli score \
  --resume data/resumes/sample_resume.txt \
  --jd     data/job_descriptions/sample_jd.txt \
  --out    outputs/sample_report.json

# Get a full faithful explanation: score breakdown, semantic evidence,
# token attributions, audit attributions, and counterfactual "what-ifs"
PYTHONPATH=src python -m resumemax.cli explain \
  --resume data/resumes/sample_resume.txt \
  --jd     data/job_descriptions/sample_jd.txt \
  --out    outputs/explanation.json
```

See [docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md) for the full
catalog of explanation methods and faithfulness guarantees.

## Streamlit UI

```bash
PYTHONPATH=src streamlit run src/resumemax/app.py
```

Open http://localhost:8501. Upload a resume, paste a JD, click "Score resume",
read the recommendations, edit your resume, and re-score. The revision history
chart tracks score uplift across iterations.

## Synthetic data generation

Generate matched **bad** / **optimized** resume pairs for any JD (deterministic
template by default; optional LLM backend if `RESUMEMAX_LLM=openai` and
`OPENAI_API_KEY` are set):

```bash
PYTHONPATH=src python -m resumemax.synthetic \
  --jd  data/gold_standard/job_descriptions \
  --out outputs/synthetic_pairs \
  --backend template --seed 7
```

## Gold-standard correlation

Validate that ResumeMax scores correlate with expert human ratings on the
shipped 8-pair pilot set (see [docs/GOLD_STANDARD.md](docs/GOLD_STANDARD.md)
for the annotation rubric):

```bash
PYTHONPATH=src python -m resumemax.evaluate gold \
  --gold data/gold_standard/gold_pairs.csv \
  --out  outputs/gold_correlation.json
```

Current pilot result: **Pearson r = 0.96, Spearman ρ = 0.88, MAE = 15.3**.

## Latency benchmark

```bash
PYTHONPATH=src python scripts/bench_latency.py --warmup --assert-budget 3.0
```

Prints a per-stage breakdown and exits non-zero if the < 3 s end-to-end
budget is violated.  See [docs/LATENCY.md](docs/LATENCY.md) for the budget
table and applied optimizations (embedding cache, batching, LLM timeout
with deterministic fallback, mock-by-default backend).

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```
