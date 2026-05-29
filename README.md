# QuantProbe

**QuantProbe: A Multi-Dimensional Framework for Probing Compression-Induced 
Hallucination and Reliability Degradation in Small Language Models**

Samuel Stephen, R. Vignesh  
Karunya Institute of Technology and Sciences, Coimbatore, India  
📧 samuels24@karunya.edu.in

---

## Overview

QuantProbe is a systematic evaluation framework that measures how weight 
quantization (FP16 → 8-bit → 4-bit) affects the reliability of small language 
models — not just accuracy, but hallucination rate, overconfidence, abstention, 
and calibration error jointly.

We introduce:
- **QuantProbe-Bench** — a 400-question benchmark designed to expose 
  compression-induced failure modes
- **Compression Reliability Score (CRS)** — a composite metric combining 
  four reliability dimensions into a single deployment-ready score

---

## Models Evaluated

| Model | Parameters | Precisions |
|---|---|---|
| TinyLlama-1.1B-Chat | 1.1B | FP16, 8-bit, 4-bit |
| Qwen 2.5-1.5B-Instruct | 1.5B | FP16, 8-bit, 4-bit |
| Phi-2 | 2.7B | FP16, 8-bit, 4-bit |

---

## Key Results

| Model | FP16 HR% | 4-bit HR% | CRS (FP16) | CRS (4-bit) |
|---|---|---|---|---|
| TinyLlama 1.1B | 38.4 | 40.6 | 0.668 | 0.655 |
| Qwen 2.5 1.5B | 26.4 | 38.9 | 0.681 | 0.654 |
| Phi-2 2.7B | 29.6 | 32.5 | 0.692 | 0.683 |

Qwen 2.5 shows the only statistically significant compression sensitivity  
(+12.6 pp, p < 0.001, 95% CI [9.3%, 16.0%]).

---

```
QuantProbe/
├── code/
│   └── evaluate.py          # Main evaluation script
├── data/
│   ├── quantprobe_bench.json  # QuantProbe-Bench (400 questions)
│   └── results/             # Raw inference outputs (9 JSON files)
├── summaries/               # Excel summary sheets
├── figures/                 # All paper figures (PNG)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/samuelstephen77/QuantProbe
cd QuantProbe
pip install -r requirements.txt
```

---

## Running the Evaluation

```bash
python code/evaluate.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --precision fp16 \
  --datasets truthfulqa freshqa quantprobe_bench \
  --output data/results/
```

Supported precision values: `fp16`, `8bit`, `4bit`

---

## Datasets

| Dataset | Source |
|---|---|
| TruthfulQA | https://github.com/sylinrl/TruthfulQA |
| FreshQA | https://github.com/freshllms/freshqa |
| QuantProbe-Bench | This repo — `data/quantprobe_bench.json` |

> Note: FaithDial was included in the pipeline but excluded from all 
> reported results due to degenerate empty-prompt outputs for all three 
> models. See paper Section 5 for details.

---

## Requirements

```
torch>=2.2.0
transformers>=4.40.0
bitsandbytes>=0.46.0
numpy>=1.24.0
scipy>=1.11.0
datasets>=2.14.0
```


## Citation

If you use QuantProbe or QuantProbe-Bench in your research, please cite:

```bibtex
@article{stephen2026quantprobe,
  title={QuantProbe: A Multi-Dimensional Framework for Probing 
         Compression-Induced Hallucination and Reliability Degradation 
         in Small Language Models},
  author={Stephen, Samuel and Vignesh, R.},
  journal={Intelligent Systems with Applications},
  year={2026},
  institution={Karunya Institute of Technology and Sciences}
}
```

---

## Data Availability

Raw results archived at:  
https://doi.org/10.5281/zenodo.XXXXXXX

---

## License

MIT License — see LICENSE file.
