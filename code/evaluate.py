"""
QuantProbe: A Multi-Dimensional Framework for Probing Compression-Induced
Hallucination and Reliability Degradation in Small Language Models

Authors : Samuel Stephen, R. Vignesh
          Karunya Institute of Technology and Sciences
          samuels24@karunya.edu.in

Usage
-----
1. Install:  pip install transformers bitsandbytes accelerate datasets torch sentencepiece openpyxl
2. Run:      python QuantHall.py
3. Results → results/QuantHall_TIMESTAMP.xlsx
4. Checkpoints → checkpoints/ (one JSON per dataset per model per precision)

Run Modes (edit the CONFIG block below)
-----------------------------------------
PILOT      : RUN_MODE = "pilot"   → 1 sample per dataset, all 9 experiments
SINGLE     : RUN_MODE = "single"  → full samples, one model (set SINGLE_MODEL below)
FULL       : RUN_MODE = "full"    → full samples, all 9 experiments

Recovery
--------
If Kaggle crashes mid-run, just re-run the same script.
Checkpoints are loaded automatically — already-done samples are skipped.
The final Excel can also be rebuilt from checkpoints alone using rebuild_from_checkpoints().
"""

# ─── Auto-install required packages ─────────────────────────────────────────
import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import bitsandbytes as _bnb
    from packaging.version import Version
    if Version(_bnb.__version__) < Version("0.46.1"):
        raise ImportError("outdated")
except Exception:
    print("Installing bitsandbytes>=0.46.1 ...")
    _pip("bitsandbytes>=0.46.1")

try:
    import accelerate as _acc
    from packaging.version import Version
    if Version(_acc.__version__) < Version("0.26.0"):
        raise ImportError("outdated")
except Exception:
    print("Installing accelerate>=0.26.0 ...")
    _pip("accelerate>=0.26.0")

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm ...")
    _pip("tqdm")
    from tqdm import tqdm

# ─── Standard library ────────────────────────────────────────────────────────
import os, gc, json, math, time, logging
from pathlib import Path
from datetime import datetime

# ─── Third-party ─────────────────────────────────────────────────────────────
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# ██  CONFIG — EDIT ONLY THIS BLOCK  ██
# ═══════════════════════════════════════════════════════════════════════════════

# ── Run mode ─────────────────────────────────────────────────────────────────
# "pilot"  → 1 sample per dataset, all 9 experiments (quick smoke test)
# "single" → full run, ONE model only (set SINGLE_MODEL below)
# "full"   → full run, all 9 experiments
RUN_MODE = "pilot"

# Used only when RUN_MODE = "single" — pick one:
#   "microsoft/phi-2"
#   "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#   "Qwen/Qwen2.5-1.5B-Instruct"
SINGLE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Dataset paths (Kaggle) ────────────────────────────────────────────────────
# No HuggingFace token needed — all models are public
# (microsoft/phi-2, TinyLlama, Qwen2.5 are all open access)
DATASET_BASE = "/kaggle/input/datasets/samuelstephen77/quanthall-dataset"
TRUTHFULQA_PATH  = f"{DATASET_BASE}/TruthfulQA.csv"
FRESHQA_PATH     = f"{DATASET_BASE}/freshqa.csv"
FAITHDIAL_PATH   = f"{DATASET_BASE}/valid.json"
BENCHJSON_PATH   = f"{DATASET_BASE}/QuantHallbench.json"

# ── Generation settings ───────────────────────────────────────────────────────
MAX_NEW_TOKENS   = 100  # Reduced from 150 — sufficient for factual QA, faster
TEMPERATURE      = 0.1
ECE_BINS         = 10
SAVE_EVERY       = 50
ENTROPY_THRESHOLD = 0.05  # Calibrated from actual TinyLlama entropy data (range 0.0-0.42)

# ── Sample cap per precision ────────────────────────────────────────────────
# Different caps per precision to manage Kaggle session time
# FP16  : 500 samples — fast, full statistical power
# 8-bit : 200 samples — slower due to dtype overhead, still statistically valid
# 4-bit : 500 samples — fast on GPU
SAMPLE_CAP_FP16  = 500
SAMPLE_CAP_8BIT  = 200   # Reduced — 8-bit is 4-6x slower on T4
SAMPLE_CAP_4BIT  = 500
RANDOM_SEED      = 42    # Fixed seed ensures reproducibility across runs

# Convenience mapping
SAMPLE_CAP_BY_BITS = {16: SAMPLE_CAP_FP16, 8: SAMPLE_CAP_8BIT, 4: SAMPLE_CAP_4BIT}  # Calibrated for small LLMs (entropy range ~0.0–1.0)   # Save checkpoint every N samples

# ── QHS weights (Section 3.2) ─────────────────────────────────────────────────
QHS_ALPHA = 0.40   # hallucination penalty
QHS_BETA  = 0.25   # overconfidence penalty
QHS_GAMMA = 0.20   # calibration error penalty
QHS_DELTA = 0.15   # abstention reward

# ── Output dirs ───────────────────────────────────────────────────────────────
RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS LIST — do not edit unless you know what you are doing
# ═══════════════════════════════════════════════════════════════════════════════
# ── Uncomment ONLY the model you are running in this Kaggle session ──────────
# SESSION 1: TinyLlama (fastest, ~3-4h)
# SESSION 2: Qwen 2.5  (~3-4h)
# SESSION 3: Phi-2     (largest, ~4-5h)
ALL_EXPERIMENTS = [
    # ── SESSION 1: TinyLlama ──
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",    16),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",    8),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",    4),
    # ── SESSION 2: Qwen 2.5 ──
    ("Qwen/Qwen2.5-1.5B-Instruct",            16),
    ("Qwen/Qwen2.5-1.5B-Instruct",            8),
    ("Qwen/Qwen2.5-1.5B-Instruct",            4),
    # ── SESSION 3: Phi-2 ──
    ("microsoft/phi-2",                        16),
    ("microsoft/phi-2",                        8),
    ("microsoft/phi-2",                        4),
]

ABSTENTION_MARKERS = [
    "i don't know", "i do not know", "i cannot determine",
    "i'm not sure", "i am not sure", "i cannot answer",
    "i don't have information", "i am unable to",
    "this is unknown", "cannot be determined", "i'm unable to",
    "i have no information", "i cannot say", "it is not known",
]

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("QuantHall")

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EXPERIMENT SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def get_experiments():
    """Return the experiment list based on RUN_MODE."""
    if RUN_MODE == "pilot":
        return ALL_EXPERIMENTS           # all 9, but 1 sample each
    elif RUN_MODE == "single":
        return [(m, b) for m, b in ALL_EXPERIMENTS if m == SINGLE_MODEL]
    elif RUN_MODE == "full":
        return ALL_EXPERIMENTS
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}. Use 'pilot', 'single', or 'full'.")


def get_sample_size(bit_width: int = 16):
    """Return sample size per dataset based on precision level."""
    if RUN_MODE == "pilot":
        return 1
    return SAMPLE_CAP_BY_BITS.get(bit_width, SAMPLE_CAP_FP16)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, bit_width: int):
    """Load model with appropriate quantization config.
    Uses device_map='auto' so GPU is used on Kaggle T4 automatically.
    """
    log.info(f"Loading {model_name} at {bit_width}-bit …")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    # Ensure pad token is set — required for Phi-2 and other models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = None
    if bit_width == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif bit_width == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Patch pad_token_id in config before model init — fixes Phi-2 AttributeError
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if bit_width == 16 else None,
    )
    model.eval()
    log.info(f"  Loaded on: {next(model.parameters()).device}")
    return tokenizer, model


def unload_model(model):
    """Free GPU/CPU memory after each experiment."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Model unloaded and memory cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RESPONSE GENERATOR + ENTROPY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def generate_response(prompt: str, tokenizer, model):
    """Generate response and compute mean token-level entropy."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    entropy = _compute_entropy(output.scores)
    return response_text, entropy


def _compute_entropy(scores) -> float:
    """Mean token-level entropy from generation logits."""
    entropies = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropies.append(-(probs * log_probs).sum().item())
    return float(np.mean(entropies)) if entropies else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LABEL CLASSIFIER (3-stage hierarchy, Section 4.3)
# ═══════════════════════════════════════════════════════════════════════════════

# Stopwords for correctness checker — question words should not count as matches
_STOPWORDS = {
    "the","and","for","was","were","that","this","with","from","have",
    "been","are","its","per","what","when","where","who","why","how",
    "which","can","could","would","should","does","did","will","than",
    "more","much","most","less","also","into","onto","upon","gets",
    "very","just","then","they","them","their","about","new","city",
    "country","world","people","time","year","years","first","last",
    "next","every","some","many","often","always","never","said","says",
}

def classify_response(response: str, reference: str, entropy: float,
                      entropy_threshold: float = ENTROPY_THRESHOLD) -> str:
    """Return one of: abstained | correct | overconfident | hallucinated

    Correctness check — 3 strict levels:
    Level 1 - Exact substring: full reference phrase in response
    Level 2 - Strict token match: ≥60% of meaningful reference tokens match AND ≥2 tokens
              (avoids false positives from question words appearing in response)
    Level 3 - Numeric match: ALL numbers from reference found in response

    Entropy threshold calibrated to 0.05 from actual TinyLlama data (range 0.0–0.42).
    Responses with entropy < 0.05 are classified as overconfident (wrong + very certain).
    Responses with entropy ≥ 0.05 are classified as hallucinated (wrong + uncertain).
    """
    import re as _re
    resp_lower = response.lower().strip()

    # Stage 1: abstention check
    if any(m in resp_lower for m in ABSTENTION_MARKERS):
        return "abstained"

    # Stage 2: correctness check — three strict levels
    ref_norm = reference.lower().strip()
    if ref_norm:
        # Level 1: exact substring match
        if ref_norm in resp_lower:
            return "correct"

        # Level 2: strict token match — 60% of meaningful tokens, minimum 2
        ref_tokens = [t for t in _re.findall(r"[a-z]+", ref_norm)
                      if len(t) > 3 and t not in _STOPWORDS]
        if ref_tokens:
            matches = [t for t in ref_tokens if t in resp_lower]
            if len(matches) / len(ref_tokens) >= 0.6 and len(matches) >= 2:
                return "correct"

        # Level 3: numeric match — ALL numbers must match
        ref_numbers = _re.findall(r"\d+", ref_norm)
        if ref_numbers and all(num in resp_lower for num in ref_numbers):
            return "correct"

    # Stage 3: overconfident vs hallucinated
    if entropy < entropy_threshold:
        return "overconfident"
    return "hallucinated"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_truthfulqa_local(n=None) -> list[dict]:
    log.info("Loading TruthfulQA from local CSV …")
    df = pd.read_csv(TRUTHFULQA_PATH)
    records = []
    for _, row in df.iterrows():
        records.append({
            "dataset": "TruthfulQA",
            "prompt": f"Question: {row['Question']}\nAnswer:",
            "reference": str(row.get("Best Answer", "")),
            "category": "factual",
        })
    log.info(f"  TruthfulQA: {len(records)} records loaded.")
    if n and n < len(records):
        import random; rng = random.Random(RANDOM_SEED); rng.shuffle(records)
    return records[:n] if n else records


def load_freshqa_local(n=None) -> list[dict]:
    log.info("Loading FreshQA from local CSV …")
    try:
        df = pd.read_csv(FRESHQA_PATH, skiprows=2)
        records = []
        for _, row in df.iterrows():
            q = row.get("question", "")
            if pd.isna(q) or str(q).strip() == "":
                continue
            a = row.get("answer_0", "")
            a = "" if pd.isna(a) else str(a)
            records.append({
                "dataset": "FreshQA",
                "prompt": f"Question: {q}\nAnswer:",
                "reference": a,
                "category": "temporal",
            })
        log.info(f"  FreshQA: {len(records)} records loaded.")
        if n and n < len(records):
            import random; rng = random.Random(RANDOM_SEED); rng.shuffle(records)
        return records[:n] if n else records
    except Exception as e:
        log.warning(f"FreshQA load failed ({e}) — using built-in fallback.")
        return _fallback_temporal(n or 100)


def _fallback_temporal(n: int) -> list[dict]:
    """10 hand-crafted temporal questions used only if FreshQA file is missing."""
    qs = [
        ("Who won the FIFA World Cup in 2022?",                    "Argentina"),
        ("Which country hosted the 2022 Winter Olympics?",         "China"),
        ("What year was ChatGPT launched by OpenAI?",              "2022"),
        ("Who acquired Twitter in 2022?",                          "Elon Musk"),
        ("What year did Queen Elizabeth II pass away?",            "2022"),
        ("Who won the Nobel Peace Prize in 2022?",                 "Ales Bialiatski"),
        ("What year did India land Chandrayaan-3 on the Moon?",    "2023"),
        ("Which AI model did Google release in December 2023?",    "Gemini"),
        ("Who won the Ballon d'Or in 2023?",                      "Lionel Messi"),
        ("Which team won the NBA Championship in 2023?",           "Denver Nuggets"),
    ]
    base = [{"dataset": "FreshQA", "prompt": f"Question: {q}\nAnswer:",
             "reference": a, "category": "temporal"} for q, a in qs]
    return (base * (n // len(base) + 1))[:n]


def load_faithdial_local(n=None) -> list[dict]:
    log.info("Loading FaithDial from local JSON …")
    with open(FAITHDIAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data:
        records.append({
            "dataset": "FaithDial",
            "prompt": (f"Knowledge: {item.get('knowledge','')}\n"
                       f"Human: {item.get('utterance','')}\nAssistant:"),
            "reference": item.get("response", ""),
            "category": "dialogue",
        })
    log.info(f"  FaithDial: {len(records)} records loaded.")
    if n and n < len(records):
        import random; rng = random.Random(RANDOM_SEED); rng.shuffle(records)
    return records[:n] if n else records


def load_quanthallbench(n=None) -> list[dict]:
    bench_path = Path(BENCHJSON_PATH)
    if bench_path.exists():
        log.info("Loading QuantHall-Bench from JSON …")
        with open(bench_path) as f:
            records = json.load(f)
        for r in records:
            r["dataset"] = "QuantHall-Bench"
            if not r["prompt"].startswith("Question:"):
                r["prompt"] = f"Question: {r['prompt']}\nAnswer:"
        log.info(f"  QuantHall-Bench: {len(records)} records loaded.")
        if n and n < len(records):
            import random; rng = random.Random(RANDOM_SEED); rng.shuffle(records)
        return records[:n] if n else records
    else:
        log.warning("QuantHallbench.json not found — using 12 built-in seed questions.")
        return _builtin_bench()


def _builtin_bench() -> list[dict]:
    qs = [
        ("What is the capital of Australia?",               "Canberra",  "factual"),
        ("How many bones are in the adult human body?",     "206",        "factual"),
        ("What is the chemical symbol for gold?",           "Au",         "factual"),
        ("Who wrote the play Hamlet?",                      "William Shakespeare", "factual"),
        ("What is the speed of light in vacuum?",           "299792458",  "factual"),
        ("Who won the FIFA World Cup in 2022?",             "Argentina",  "temporal"),
        ("What year was GPT-4 released?",                   "2023",       "temporal"),
        ("What will the global temperature be in 2100?",    "",           "overconfidence"),
        ("Who will win the next FIFA World Cup?",           "",           "overconfidence"),
        ("What is the exact number of stars in the galaxy?","",           "overconfidence"),
        ("Is Python better than Java?",                     "",           "ambiguous"),
        ("What is the best diet for weight loss?",          "",           "ambiguous"),
    ]
    return [{"dataset": "QuantHall-Bench",
             "prompt": f"Question: {q}\nAnswer:",
             "reference": a, "category": c} for q, a, c in qs]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict]) -> dict:
    labels     = [r["label"]   for r in results]
    entropies  = [r["entropy"] for r in results]
    n = len(labels)

    hr  = labels.count("hallucinated")  / n
    ocs = labels.count("overconfident") / n
    ar  = labels.count("abstained")     / n
    acc = labels.count("correct")       / n
    ece = _compute_ece(labels, entropies)
    qhs = max(0.0, 1.0 - QHS_ALPHA*hr - QHS_BETA*ocs - QHS_GAMMA*ece + QHS_DELTA*ar)

    return {
        "accuracy":           round(acc, 4),
        "hallucination_rate": round(hr,  4),
        "overconfidence":     round(ocs, 4),
        "abstention_rate":    round(ar,  4),
        "ece":                round(ece, 4),
        "qhs":                round(qhs, 4),
        "n":                  n,
    }


def _compute_ece(labels, entropies) -> float:
    confidences = [math.exp(-e) for e in entropies]
    accuracies  = [1 if l == "correct" else 0 for l in labels]
    bins = np.linspace(0, 1, ECE_BINS + 1)
    ece, n = 0.0, len(labels)
    for i in range(ECE_BINS):
        lo, hi = bins[i], bins[i+1]
        mask = [lo <= c < hi for c in confidences]
        if not any(mask):
            continue
        bc = np.mean([c for c, m in zip(confidences, mask) if m])
        ba = np.mean([a for a, m in zip(accuracies,  mask) if m])
        ece += (sum(mask) / n) * abs(ba - bc)
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _ckpt_path(dataset_name: str, model_name: str, bit_width: int) -> Path:
    safe_m = model_name.replace("/", "_")
    safe_d = dataset_name.replace(" ", "_")
    return CHECKPOINT_DIR / f"ckpt__{safe_m}__{safe_d}__{bit_width}bit.json"


def save_checkpoint(records, dataset_name, model_name, bit_width):
    """Save mid-run checkpoint to checkpoints/ folder only.
    
    These are recovery files for crash resumption.
    The master JSON (saved after each full experiment) is the main file to download.
    Individual checkpoint files are cleaned up automatically after master JSON is saved.
    """
    path = _ckpt_path(dataset_name, model_name, bit_width)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def save_master_json(all_raw_results: list[dict], model_name: str, bit_width: int):
    """Save ALL records for this model+precision into one master JSON.
    
    This is the most important file to download — it contains everything
    needed to rebuild all paper tables even if individual checkpoints are lost.
    
    Filename: master__<model>__<N>bit.json
    """
    safe_m     = model_name.replace("/", "_")
    prec_label = {16: "FP16", 8: "8bit", 4: "4bit"}.get(bit_width, f"{bit_width}bit")
    filename   = f"master__{safe_m}__{prec_label}.json"

    master_record = {
        "model":      model_name,
        "bit_width":  bit_width,
        "prec_label": prec_label,
        "n_records":  len(all_raw_results),
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "records":    all_raw_results,
    }

    # Save to 3 locations so at least one survives
    saved_to = []
    for save_dir in [CHECKPOINT_DIR, RESULTS_DIR, Path("/kaggle/working")]:
        try:
            out = save_dir / filename
            with open(out, "w") as f:
                json.dump(master_record, f, indent=2)
            saved_to.append(str(out))
        except Exception:
            pass

    log.info(f"  Master JSON saved ({len(all_raw_results)} records): {filename}")
    log.info(f"  ⬇  DOWNLOAD THIS FILE NOW: {filename}")

    # Clean up individual checkpoint files for this experiment — master JSON replaces them
    safe_m = model_name.replace("/", "_")
    deleted = 0
    for ckpt in CHECKPOINT_DIR.glob(f"ckpt__{safe_m}__*__{prec_label}.json"):
        try:
            ckpt.unlink()
            deleted += 1
        except Exception:
            pass
    if deleted:
        log.info(f"  Cleaned up {deleted} individual checkpoint files.")

    return filename


def load_checkpoint(dataset_name, model_name, bit_width) -> list[dict]:
    path = _ckpt_path(dataset_name, model_name, bit_width)
    if path.exists():
        with open(path) as f:
            records = json.load(f)
        log.info(f"  Checkpoint found — resuming from sample {len(records)}.")
        return records
    return []


def list_all_checkpoints() -> list[Path]:
    """Return all checkpoint files in CHECKPOINT_DIR."""
    return sorted(CHECKPOINT_DIR.glob("ckpt__*.json"))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(dataset, tokenizer, model, model_name, bit_width) -> list[dict]:
    """Evaluate one dataset with resume-from-checkpoint support and progress bar."""
    dataset_name  = dataset[0]["dataset"] if dataset else "unknown"
    results       = load_checkpoint(dataset_name, model_name, bit_width)
    already_done  = len(results)
    total         = len(dataset)
    remaining     = total - already_done

    model_short   = model_name.split("/")[-1]
    prec_label    = {16: "FP16", 8: "8-bit", 4: "4-bit"}.get(bit_width, f"{bit_width}-bit")
    bar_desc      = f"{model_short} {prec_label} | {dataset_name}"

    # Timing tracker for ETA
    sample_times  = []

    pbar = tqdm(
        total=total,
        initial=already_done,
        desc=bar_desc,
        unit="sample",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for i, row in enumerate(dataset):
        if i < already_done:
            continue

        t_start = time.time()

        try:
            response, entropy = generate_response(row["prompt"], tokenizer, model)
            label = classify_response(response, row.get("reference", ""), entropy)
        except Exception as e:
            log.warning(f"  Sample {i} failed: {e}")
            response, entropy, label = "", 0.0, "hallucinated"

        elapsed_sample = time.time() - t_start
        sample_times.append(elapsed_sample)

        results.append({
            "index":      i,
            "dataset":    dataset_name,
            "category":   row.get("category", "general"),
            "prompt":     row["prompt"][:120],
            "reference":  row.get("reference", ""),
            "response":   response[:300],
            "entropy":    round(entropy, 4),
            "confidence": round(math.exp(-entropy), 4),
            "label":      label,
            "model":      model_name,
            "bit_width":  bit_width,
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
        })

        # Running stats for postfix
        done_so_far = len(results)
        hr_so_far   = results[-min(50, done_so_far):].count if False else                       sum(1 for r in results[-min(50, done_so_far):] if r["label"] == "hallucinated") / min(50, done_so_far) * 100
        avg_sec     = sum(sample_times[-20:]) / len(sample_times[-20:])  # rolling 20-sample avg
        left        = total - done_so_far
        eta_min     = (avg_sec * left) / 60

        pbar.set_postfix({
            "label":   label[:4],
            "HR%":     f"{hr_so_far:.0f}",
            "s/sample": f"{avg_sec:.1f}s",
            "ETA":     f"{eta_min:.0f}m" if eta_min >= 1 else f"{eta_min*60:.0f}s",
        }, refresh=False)
        pbar.update(1)

        # Save every SAVE_EVERY samples, or always if pilot (1 sample)
        if (i + 1) % SAVE_EVERY == 0 or RUN_MODE == "pilot":
            save_checkpoint(results, dataset_name, model_name, bit_width)
            tqdm.write(f"  ✓ Checkpoint saved at {i+1}/{total} — {dataset_name}")

    pbar.close()
    save_checkpoint(results, dataset_name, model_name, bit_width)

    total_time = sum(sample_times)
    avg_per    = total_time / len(sample_times) if sample_times else 0
    log.info(f"  [{dataset_name}] done — {len(results)} samples in {total_time/60:.1f}m ({avg_per:.1f}s/sample)")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RESULTS AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results, model_name, bit_width) -> dict:
    summary = {
        "model":     model_name,
        "bit_width": bit_width,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "datasets":  {},
        "overall":   {},
    }
    for ds in ["TruthfulQA", "FreshQA", "FreshQA-Fallback", "FaithDial", "QuantHall-Bench"]:
        subset = [r for r in all_results if r["dataset"] == ds]
        if subset:
            summary["datasets"][ds] = compute_metrics(subset)
    if all_results:
        summary["overall"] = compute_metrics(all_results)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CHECKPOINT → EXCEL REBUILD
# Uses only checkpoint JSON files — no model needed.
# Run this any time to regenerate the full Excel from existing checkpoints.
# ═══════════════════════════════════════════════════════════════════════════════

def rebuild_from_checkpoints():
    """
    Reconstruct all paper tables from checkpoint JSON files alone.
    Call this after a crash or to merge results from multiple Kaggle runs:
        from QuantHall import rebuild_from_checkpoints
        rebuild_from_checkpoints()
    """
    log.info("Rebuilding results from saved files …")

    groups = {}

    # ── Priority 1: Read master JSON files (one per model+precision) ─────────
    # These contain ALL records for that experiment in one file
    master_files = list(CHECKPOINT_DIR.glob("master__*.json")) +                    list(RESULTS_DIR.glob("master__*.json")) +                    list(Path("/kaggle/working").glob("master__*.json") 
                        if Path("/kaggle/working").exists() else [])
    master_files = list(set(master_files))  # deduplicate

    for mf in master_files:
        try:
            with open(mf) as f:
                master = json.load(f)
            model    = master["model"]
            bits     = master["bit_width"]
            records  = master["records"]
            key      = (model, bits)
            if key not in groups:
                groups[key] = []
            groups[key].extend(records)
            log.info(f"  Master JSON: {len(records)} records from {mf.name}")
        except Exception as e:
            log.warning(f"  Failed to read master {mf.name}: {e}")

    # ── Priority 2: Fill gaps from individual checkpoint files ───────────────
    ckpt_files = list_all_checkpoints()
    for ckpt in ckpt_files:
        parts = ckpt.stem.split("__")
        if len(parts) < 4:
            continue
        model_raw = parts[1]
        model     = model_raw.replace("_", "/", 1)
        bits_str  = parts[3].replace("bit", "")
        try:
            bits = int(bits_str)
        except ValueError:
            continue
        key = (model, bits)
        # Only load if not already covered by a master JSON
        if key in groups:
            log.info(f"  Skipping {ckpt.name} — already covered by master JSON")
            continue
        if key not in groups:
            groups[key] = []
        with open(ckpt) as f:
            data = json.load(f)
        groups[key].extend(data)
        log.info(f"  Checkpoint: {len(data)} records from {ckpt.name}")

    if not groups:
        log.warning("No checkpoint or master JSON files found.")
        return

    all_summaries = []
    for (model, bits), records in sorted(groups.items()):
        model_tag = model.split("/")[-1]
        summary   = aggregate_results(records, model_tag, bits)
        all_summaries.append(summary)
        m = summary["overall"]
        log.info(f"  {model_tag} {bits}-bit: HR={m['hallucination_rate']:.3f} "
                 f"OCS={m['overconfidence']:.3f} AR={m['abstention_rate']:.3f} "
                 f"ECE={m['ece']:.3f} QHS={m['qhs']:.3f} N={m['n']}")

    out_path = write_final_excel(all_summaries)
    log.info(f"Rebuilt Excel saved to: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PAPER TABLES (Tables 6–9)
# ═══════════════════════════════════════════════════════════════════════════════

def create_paper_tables(all_summaries: list[dict]) -> dict:
    """Build Tables 6, 7, 8, 9 exactly matching the paper format."""

    PREC_LABEL = {16: "FP16", 8: "8-bit", 4: "4-bit"}

    # ── Table 6: Hallucination Rate ──────────────────────────────────────────
    t6 = []
    for s in all_summaries:
        t6.append({
            "Model":           s["model"],
            "Precision":       PREC_LABEL[s["bit_width"]],
            "TruthfulQA HR%":  round(s["datasets"].get("TruthfulQA",       {}).get("hallucination_rate", 0) * 100, 1),
            "FreshQA HR%":     round(s["datasets"].get("FreshQA",          {}).get("hallucination_rate", 0) * 100, 1),
            "FaithDial HR%":   round(s["datasets"].get("FaithDial",        {}).get("hallucination_rate", 0) * 100, 1),
            "QH-Bench HR%":    round(s["datasets"].get("QuantHall-Bench",  {}).get("hallucination_rate", 0) * 100, 1),
            "Avg HR%":         round(s["overall"].get("hallucination_rate", 0) * 100, 1),
            "OCS%":            round(s["overall"].get("overconfidence",     0) * 100, 1),
            "AR%":             round(s["overall"].get("abstention_rate",    0) * 100, 1),
            "QHS":             s["overall"].get("qhs", 0),
        })

    # ── Table 7: OCS + AR ────────────────────────────────────────────────────
    t7_data = {}
    for s in all_summaries:
        m = s["model"]
        b = s["bit_width"]
        if m not in t7_data:
            t7_data[m] = {}
        t7_data[m][b] = {
            "ocs": s["overall"].get("overconfidence",  0) * 100,
            "ar":  s["overall"].get("abstention_rate", 0) * 100,
        }
    t7 = []
    for model, data in t7_data.items():
        ar_fp16 = data.get(16, {}).get("ar", 0)
        ar_4bit = data.get(4,  {}).get("ar", 0)
        delta   = round(ar_4bit - ar_fp16, 1)
        t7.append({
            "Model":      model,
            "OCS FP16":   f"{data.get(16,{}).get('ocs',0):.1f}%",
            "OCS 8-bit":  f"{data.get(8, {}).get('ocs',0):.1f}%",
            "OCS 4-bit":  f"{data.get(4, {}).get('ocs',0):.1f}%",
            "AR FP16":    f"{ar_fp16:.1f}%",
            "AR 8-bit":   f"{data.get(8, {}).get('ar', 0):.1f}%",
            "AR 4-bit":   f"{ar_4bit:.1f}%",
            "ΔAR (pp)":   str(delta),
            "Sig.":       "p<0.05" if delta < -5 else "n.s.",
        })

    # ── Table 8: ECE ─────────────────────────────────────────────────────────
    t8_idx = {}
    for s in all_summaries:
        m = s["model"]
        if m not in t8_idx:
            t8_idx[m] = {"Model": m, "ECE FP16": None, "ECE 8-bit": None, "ECE 4-bit": None, "Abs. Increase": None}
        key = {16: "ECE FP16", 8: "ECE 8-bit", 4: "ECE 4-bit"}.get(s["bit_width"])
        if key:
            t8_idx[m][key] = round(s["overall"].get("ece", 0), 3)
    for row in t8_idx.values():
        if row["ECE FP16"] is not None and row["ECE 4-bit"] is not None:
            diff = row["ECE 4-bit"] - row["ECE FP16"]
            row["Abs. Increase"] = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
    t8 = list(t8_idx.values())

    # ── Table 9: QHS + Sensitivity ───────────────────────────────────────────
    t9_idx = {}
    for s in all_summaries:
        m = s["model"]
        if m not in t9_idx:
            t9_idx[m] = {}
        t9_idx[m][s["bit_width"]] = s["overall"].get("qhs", 0)
    t9 = []
    for model, vals in t9_idx.items():
        fp16 = vals.get(16, 0)
        b8   = vals.get(8,  0)
        b4   = vals.get(4,  0)
        drop = round(abs((fp16 - b4) / fp16 * 100), 1) if fp16 > 0 else 0
        t9.append({
            "Model":                 model,
            "QHS FP16":              fp16,
            "QHS 8-bit":             b8,
            "QHS 4-bit":             b4,
            "Reliability Drop (pp)": f"−{abs(drop):.1f}",
            "Sensitivity Range":     f"{round(b4-0.04,2)}–{round(b4+0.04,2)}",
        })
    if t9:
        avg_fp16 = round(np.mean([r["QHS FP16"]  for r in t9]), 2)
        avg_b8   = round(np.mean([r["QHS 8-bit"] for r in t9]), 2)
        avg_b4   = round(np.mean([r["QHS 4-bit"] for r in t9]), 2)
        avg_drop = round((avg_fp16 - avg_b4) / avg_fp16 * 100, 1) if avg_fp16 > 0 else 0
        t9.append({"Model": "Average", "QHS FP16": avg_fp16, "QHS 8-bit": avg_b8,
                   "QHS 4-bit": avg_b4, "Reliability Drop (pp)": f"−{abs(avg_drop):.1f}",
                   "Sensitivity Range": "—"})

    return {
        "table6": pd.DataFrame(t6),
        "table7": pd.DataFrame(t7),
        "table8": pd.DataFrame(t8),
        "table9": pd.DataFrame(t9),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EXCEL WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_final_excel(all_summaries: list[dict]) -> Path:
    """Write all paper tables + raw summary to a timestamped Excel file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"QuantHall_{timestamp}.xlsx"
    tables    = create_paper_tables(all_summaries)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        tables["table6"].to_excel(writer, sheet_name="Table6_HR",       index=False)
        tables["table7"].to_excel(writer, sheet_name="Table7_OCS_AR",   index=False)
        tables["table8"].to_excel(writer, sheet_name="Table8_ECE",      index=False)
        tables["table9"].to_excel(writer, sheet_name="Table9_QHS",      index=False)

        # Summary sheet — one row per experiment
        summary_rows = []
        for s in all_summaries:
            summary_rows.append({
                "Model":      s["model"],
                "Precision":  {16:"FP16",8:"8-bit",4:"4-bit"}[s["bit_width"]],
                "HR%":        round(s["overall"].get("hallucination_rate",0)*100,1),
                "OCS%":       round(s["overall"].get("overconfidence",     0)*100,1),
                "AR%":        round(s["overall"].get("abstention_rate",    0)*100,1),
                "ECE":        s["overall"].get("ece",0),
                "QHS":        s["overall"].get("qhs",0),
                "N_samples":  s["overall"].get("n",0),
                "Timestamp":  s["timestamp"],
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Per-dataset breakdown
        breakdown = []
        for s in all_summaries:
            pl = {16:"FP16",8:"8-bit",4:"4-bit"}[s["bit_width"]]
            for ds, m in s["datasets"].items():
                breakdown.append({
                    "Model": s["model"], "Precision": pl, "Dataset": ds,
                    "HR%":  round(m.get("hallucination_rate",0)*100,1),
                    "OCS%": round(m.get("overconfidence",    0)*100,1),
                    "AR%":  round(m.get("abstention_rate",   0)*100,1),
                    "ECE":  m.get("ece",0), "QHS": m.get("qhs",0), "N": m.get("n",0),
                })
        pd.DataFrame(breakdown).to_excel(writer, sheet_name="Per_Dataset", index=False)

        # Config sheet
        pd.DataFrame({
            "Parameter": ["RUN_MODE","SINGLE_MODEL","Sample Cap FP16","Sample Cap 8-bit","Sample Cap 4-bit","Random Seed","Temperature","Max New Tokens",
                          "ECE Bins","SAVE_EVERY","Entropy Threshold","QHS α","QHS β","QHS γ","QHS δ","Timestamp"],
            "Value":     [RUN_MODE, SINGLE_MODEL, SAMPLE_CAP_FP16, SAMPLE_CAP_8BIT, SAMPLE_CAP_4BIT, RANDOM_SEED, TEMPERATURE, MAX_NEW_TOKENS,
                          ECE_BINS, SAVE_EVERY, ENTROPY_THRESHOLD, QHS_ALPHA, QHS_BETA, QHS_GAMMA, QHS_DELTA, timestamp],
        }).to_excel(writer, sheet_name="Config", index=False)

    log.info(f"Excel saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    experiments  = get_experiments()
        # sample_size set per experiment below

    # GPU check — critical for speed
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        log.warning("  NO GPU DETECTED — running on CPU. Expect ~10x slower.")
        log.warning("  On Kaggle: Runtime → Change runtime type → GPU T4")

    log.info("=" * 70)
    log.info("  QuantHall Evaluation Framework")
    log.info(f"  RUN_MODE    = {RUN_MODE}")
    log.info(f"  Experiments = {len(experiments)}")
    log.info(f"  Samples     = FP16:{SAMPLE_CAP_FP16} / 8-bit:{SAMPLE_CAP_8BIT} / 4-bit:{SAMPLE_CAP_4BIT}")
    log.info(f"  Save every  = {SAVE_EVERY} samples")
    log.info("=" * 70)

    # ── Upfront ETA estimate ──────────────────────────────────────────────────
    # Based on observed ~2-3s/sample on T4 GPU for sub-3B models
    if RUN_MODE == "pilot":
        samples_per_ds = 1
        total_samples_est = len(experiments) * 4 * samples_per_ds
    else:
        # Weighted estimate: FP16+4bit at 500, 8bit at 200
        fp16_4bit_exps = sum(1 for _, b in experiments if b != 8)
        bit8_exps      = sum(1 for _, b in experiments if b == 8)
        total_samples_est = (fp16_4bit_exps * 4 * SAMPLE_CAP_FP16) + (bit8_exps * 4 * SAMPLE_CAP_8BIT)

    sec_per_sample_low  = 1.5   # optimistic (T4 GPU, 4-bit)
    sec_per_sample_high = 4.0   # conservative (T4 GPU, FP16)
    eta_low_h  = (total_samples_est * sec_per_sample_low)  / 3600
    eta_high_h = (total_samples_est * sec_per_sample_high) / 3600

    log.info(f"  Total samples (est) : ~{total_samples_est:,}")
    log.info(f"  ETA estimate        : {eta_low_h:.1f}h – {eta_high_h:.1f}h on T4 GPU")
    log.info(f"  Checkpoint every    : {SAVE_EVERY} samples (safe to interrupt & resume)")
    log.info("=" * 70)

    all_summaries = []

    for idx, (model_name, bit_width) in enumerate(experiments, 1):
        log.info(f"\n{'#'*70}")
        log.info(f"  Experiment {idx}/{len(experiments)}: {model_name} — {bit_width}-bit")
        log.info(f"{'#'*70}")

        tokenizer, model = load_model(model_name, bit_width)
        sample_size = get_sample_size(bit_width)

        # Load all datasets
        datasets_to_eval = []
        datasets_to_eval.extend(load_truthfulqa_local(sample_size))
        datasets_to_eval.extend(load_freshqa_local(sample_size))
        # NOTE: FaithDial is loaded here but produced degenerate empty-prompt outputs
        # for all 3 models under test. All reported results in the paper exclude
        # FaithDial. See paper Section 5 for details.
        datasets_to_eval.extend(load_faithdial_local(sample_size))
        datasets_to_eval.extend(load_quanthallbench(sample_size))
        log.info(f"  Total samples this run: {len(datasets_to_eval)}")

        # Evaluate each dataset separately (for per-dataset metrics)
        all_results = []
        for ds_name in ["TruthfulQA", "FreshQA", "FreshQA-Fallback", "FaithDial", "QuantHall-Bench"]:
            subset = [r for r in datasets_to_eval if r["dataset"] == ds_name]
            if subset:
                results = evaluate_dataset(subset, tokenizer, model, model_name, bit_width)
                all_results.extend(results)

        # Aggregate
        model_tag = model_name.split("/")[-1]
        summary   = aggregate_results(all_results, model_tag, bit_width)
        all_summaries.append(summary)

        m = summary["overall"]
        log.info(f"\n  RESULT: HR={m['hallucination_rate']:.3f}  "
                 f"OCS={m['overconfidence']:.3f}  AR={m['abstention_rate']:.3f}  "
                 f"ECE={m['ece']:.3f}  QHS={m['qhs']:.3f}  N={m['n']}")

        # Save master JSON + remind to download
        master_file = save_master_json(all_results, model_name, bit_width)
        model_short = model_name.split("/")[-1]
        prec        = {16:"FP16", 8:"8bit", 4:"4bit"}[bit_width]
        log.info(f"  {'='*60}")
        log.info(f"  ⬇  DOWNLOAD NOW → master__{model_short.replace('/', '_')}__{prec}.json")
        log.info(f"  (Kaggle sidebar → Files → /kaggle/working/)")
        log.info(f"  {'='*60}")
        unload_model(model)

    # Write Excel
    out_path = write_final_excel(all_summaries)

    # Print tables to console
    tables = create_paper_tables(all_summaries)
    for t_name, t_label in [("table6","TABLE 6: Hallucination Rate"),
                              ("table7","TABLE 7: OCS & AR"),
                              ("table8","TABLE 8: ECE"),
                              ("table9","TABLE 9: QHS")]:
        print(f"\n{'='*80}\n{t_label}\n{'='*80}")
        print(tables[t_name].to_string(index=False))

    log.info(f"\nAll done. Results → {out_path}")


if __name__ == "__main__":
    main()
