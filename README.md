# TIL-AI_NULL — DSTA BrainHack TIL-AI 2026

Team **NULL**'s submission for the DSTA BrainHack TIL-AI 2026 competition. This repository contains our models, inference servers and Docker setups for the four challenges: **ASR**, **CV**, **NLP** and **AE**.

> Built from the official [til-26 template](https://github.com/til-ai/til-26). Track: **[Novice / Advanced]**



## Challenges at a glance

| Challenge | Task | Overall weight | Metric |
| --------- | ---- | -------------- | ------ |
| **ASR** | Transcribe noisy speech (English, plus Malay, Tamil and Chinese on Advanced) | 20% | 1 − WER (1 − CER for Chinese) |
| **CV** | Detect and classify 18 object types (aircraft, vehicles, ships) with LTWH bounding boxes | 20% | mAP@.5:.05:.95 |
| **NLP** | Answer questions from a document corpus set in the fictional world of Clairos (RAG) | 20% | Retrieval + answer equivalence |
| **AE** | Navigate a 16x16 grid map against other agents over 200 time steps | 40% | Total reward |

Each challenge score is **75% accuracy/reward + 25% inference speed**. The speed score is `1 − min(t_elapsed, t_max) / t_max` (`t_max` = 30 minutes in Qualifiers), so faster inference scores higher.

## Our approach

### ASR (`asr/`)
- **Model:** [e.g. Whisper variant / fine-tuned model]
- **Key ideas:** [e.g. noise handling, vocabulary for fictional slang, language handling]


### CV (`cv/`)
- **Model:** [e.g. YOLO variant]; best weights selected from training runs
- **Key ideas:** [e.g. augmentation, small-object handling, LTWH conversion via `xywh2ltwh`]


### NLP (`nlp/`)
- **Pipeline:** [retriever] → [reader/generator]
- **Key ideas:** [e.g. returning top-3 document IDs, returning an empty answer for unanswerable questions]


### AE (`ae/`)
- **Agent:** [e.g. RL algorithm / heuristic policy]
- **Key ideas:** [e.g. reward shaping, handling the map]


## Repository structure

```
TIL-AI_NULL/
├── ae/                  # Autonomous exploration agent
├── asr/                 # Speech recognition
├── cv/                  # Object detection
├── nlp/                 # Question answering over documents
├── requirements-dev.txt # Development dependencies
└── README.md
```

Each challenge folder follows the same layout:

| File | Purpose |
| ---- | ------- |
| `src/` | Our code |
| `*_manager.py` | Loads the model and runs inference |
| `*_server.py` | Local web server that talks to the competition infrastructure |
| `Dockerfile` | Builds the Docker image for the model |
| `requirements.txt` | Dependencies bundled into the Docker image |
| `README.md` | Input/output format for that challenge |

## Getting started

Requires **Python 3.10+**. The competition GCP Workbench instance comes with conda installed.

```bash
# 1. Clone the repo
git clone https://github.com/<username>/TIL-AI_NULL.git
cd TIL-AI_NULL

# 2. Initialise the Git submodules
git submodule update --init

# 3. Create and activate a virtual environment
conda create --name til-asr python=3.13
conda activate til-asr

# 4. Install development dependencies
pip install -r requirements-dev.txt
```

> Do not delete or modify `til-26-finals/`, `til-26-ae/` or `.gitmodules`.

## Build, test and submit

The `til` command-line tool on the GCP Workbench instance handles all three steps:

```bash
til build asr          # build the Docker image (optional tag: til build ae my-tag)
til test asr           # run it on a local Docker network against the training data
til submit asr         # submit the image for evaluation
```

Replace `asr` with `cv`, `noise`, `nlp` or `ae`. If no tag is given, `latest` is used.


