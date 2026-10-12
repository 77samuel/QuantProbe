# QuantProbe V2

## An Exploratory Empirical Study of Reliability-Related Outcomes
## in Small Language Models under Quantization

QuantProbe is an exploratory empirical study examining
reliability-related outcomes in small language models under
different quantization configurations.

The study investigates heuristic correctness, hallucination-rate
proxies, entropy-threshold categories, abstention, an entropy-derived
uncertainty proxy, and an exploratory composite score.

The study evaluates three small language models under FP16,
8-bit, and 4-bit configurations.

This repository organizes the experimental notebook, data artifacts,
threshold analyses, and supplementary results associated with
QuantProbe V2.

**Research status:** Exploratory empirical study. The results do not
establish general quantization safety guarantees, independently
validated hallucination rates, or a validated deployment-readiness
metric.

---

## 1. Research Objectives

QuantProbe investigates the following questions:

1. How do heuristic correctness and hallucination-rate proxies vary
   across the tested model and quantization configurations?
2. How do entropy-threshold overconfidence proxies vary across
   configurations?
3. What abstention rates are observed under the tested configurations?
4. How do exploratory Composite Reliability Score (CRS) summaries
   vary, and what limitations affect their interpretation?

The study focuses on observed outcomes under the specific
experimental conditions tested.

---

## 2. Models and Experimental Design

The study evaluates the following models:

| Model | Approximate parameters |
|---|---:|
| TinyLlama 1.1B | 1.1B |
| Qwen 2.5 1.5B-Instruct | 1.5B |
| Phi-2 | 2.7B |

Three precision configurations were evaluated:

- FP16
- 8-bit using bitsandbytes LLM.int8()
- 4-bit using bitsandbytes NF4

These quantization procedures are different implementations.
Consequently, comparisons reflect both the nominal precision and
the quantization method.

### Run inventory

| Item | Reported value |
|---|---:|
| Models | 3 |
| Precision configurations | 3 |
| Random seeds | 42, 123, 7 |
| Experimental runs | 27 |
| Raw responses | 2,430 |
| Raw records per run | 90 |
| Typical scored records per run | 80 |
| Exception | One Qwen 2.5 4-bit run has 79 scored records |

Raw record counts and scored record counts are not interchangeable.

The experiments used seeded sampling. Exact deterministic
reproduction was not established.

---

## 3. Repository Structure

The intended V2 structure is:

    v2/
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    │   └── quanthall_v2.ipynb
    ├── data/
    │   ├── calibration_threshold_summary.csv
    │   ├── calibration_threshold_candidates.csv
    │   ├── QuantHallbench_withsplit.json
    │   └── raw_seeds_123_and_7.zip
    └── results/
        └── QuantProbe_Threshold_Sensitivity_Supplementary.xlsx

Only files actually uploaded to GitHub should be considered available
artifacts. The names above describe the intended organization.

Data files should be distributed only when their provenance and
licensing conditions permit redistribution.

---

## 4. Experimental Method

The evaluation pipeline includes:

1. Loading the evaluation data.
2. Loading each model under the relevant precision configuration.
3. Generating responses using the experimental settings.
4. Computing full-vocabulary predictive entropy during generation.
5. Applying the documented heuristic response-labeling procedure.
6. Calculating the study's reported metrics.
7. Comparing precision configurations using paired analyses.
8. Examining threshold and CRS weight sensitivity.

Predictive entropy is calculated from the full-vocabulary probability
distribution at each generation step and averaged over generated
tokens.

The entropy-derived confidence proxy is:

    conf(r) = exp(-H(r))

This quantity is treated as a relative uncertainty proxy, not as a
calibrated probability.

The underlying full-vocabulary probability distributions were not
retained for later calibration analysis.

---

## 5. Threshold Analysis

The manuscript reports the following exploratory,
model-specific thresholds:

| Model | Threshold |
|---|---:|
| TinyLlama | 0.100 |
| Qwen 2.5 1.5B | 0.075 |
| Phi-2 | 0.050 |

The threshold analysis uses a candidate grid of:

    0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20

Thresholds were selected using Youden's J on a retrospective
item-ID split. The manuscript reports 2,000 response-level bootstrap
replicates and 95% bootstrap confidence intervals.

The analysis remains exploratory because:

- Correctness labels are heuristic.
- The underlying records had been examined in earlier analyses.
- The split is not an untouched independent test set.
- Independent human-verified validation was not established.

The retrospective threshold analysis does not change the historical
threshold settings used in the original seed-42 summaries.
Those summaries used a global fallback threshold of 0.05.

The CSV files and supplementary workbook should be interpreted
in this context.

---

## 6. Statistical Analysis

The study reports paired correctness comparisons using exact
McNemar tests, with Holm–Bonferroni correction across 18 tests.

The manuscript reports that no comparison was statistically
significant after adjustment.

Paired bootstrap confidence intervals were also used to estimate
uncertainty in differences between mean heuristic error rates.

These analyses remain exploratory because the correctness labels
are based on heuristic matching rather than independently validated
semantic factuality judgments.

The results do not establish equivalence between precision
configurations.

---

## 7. Composite Reliability Score

The Composite Reliability Score (CRS) is retained as a secondary,
exploratory summary:

    CRS = 1 - alpha*HR - beta*OCS - gamma*EBCE + delta*AR

The reported design weights are:

| Component | Weight |
|---|---:|
| Hallucination-rate proxy (HR) | 0.40 |
| Overconfidence proxy (OCS) | 0.25 |
| Entropy-based calibration error proxy (EBCE) | 0.20 |
| Abstention rate (AR) | 0.15 |

These weights are heuristic design choices, not externally validated
parameters.

The raw score is clamped to the interval [0, 1]. The unclamped
formulation can reward excessive abstention, illustrating a
limitation of the score.

CRS is not a validated measure of factual reliability, calibration,
or deployment utility. Its sensitivity analysis evaluates numerical
variation, not construct validity.

Readers should examine individual metrics rather than relying
on CRS alone.

---

## 8. Evaluation Limitations

The findings must be interpreted in light of the following
limitations.

### Heuristic correctness labels

Correctness is assessed using lexical matching, token overlap,
numeric matching, and rule-based abstention detection.

These methods may misclassify semantically correct or incorrect
answers. Independent human factuality validation has not been
established.

### Benchmark independence

QuantProbe-Bench contains 400 questions. The available audit
identified 30 items overlapping with pilot inference records.
Independence of the remaining items has not been established.

The benchmark should be treated as a pilot resource, not an
independently validated test set.

### Threshold validation

The thresholds are exploratory estimates from previously examined
records and heuristic labels. Independent calibration and a
human-verified held-out evaluation set remain unavailable.

### Calibration proxy

EBCE is an entropy-derived proxy, not standard expected calibration
error. The study does not establish that this proxy is a calibrated
probability measure.

### Model coverage

Only three models below three billion parameters were evaluated.
The results may not generalize to other models, model families,
tasks, or languages.

### Hardware efficiency

The study reports a preliminary FP16 hardware measurement.
Comparable INT8 and INT4 efficiency measurements were not completed.
The study therefore does not establish general latency, memory,
or energy advantages for quantization.

### Reproducibility

Exact deterministic generation and exact numerical reproduction
were not established. Reproduction may depend on the model
checkpoints, package versions, hardware, and runtime configuration.

---

## 9. Reproduction Instructions

### Requirements

- Python 3.12
- PyTorch
- Hugging Face Transformers
- bitsandbytes
- The additional packages listed in `requirements.txt`
- Access to the required model checkpoints and permitted datasets

### Setup

Create and activate a Python environment appropriate for your
hardware and install the dependencies:

    pip install -r requirements.txt

Open the experimental notebook:

    notebooks/quanthall_v2.ipynb

Review the notebook configuration before running the experiments.
Confirm model access, dataset availability, device configuration,
and quantization support.

Run the notebook according to its documented execution order.

### Reproduction caveat

The presence of a notebook and supporting files does not by itself
guarantee complete reproduction. The repository should be considered
fully reproducible only after the required artifacts and execution
steps have been verified against the reported results.

---

## 10. Data Availability

The repository is intended to provide the code, data artifacts,
and supplementary analyses needed to inspect the study.

Availability depends on the files actually published in this
repository and on applicable dataset licences and access restrictions.

Readers should consult the data files and their documentation
to determine which records are included and how they were generated.

The provenance of all gold answers and the independence of all
benchmark items have not been fully established.

---

## 11. Intended Use

QuantProbe V2 is intended for research into reliability-related
outcomes in small language models under quantization.

It does not provide a guarantee of factual correctness, a validated
safety certification, or a universal ranking of deployment readiness.

The results should be interpreted as exploratory evidence from
the specific models, quantization implementations, datasets,
and experimental conditions studied.

---

## Citation

Please cite the associated QuantProbe manuscript if you use this
repository in academic work.

Complete bibliographic details should be added after the manuscript's
publication or final acceptance.
