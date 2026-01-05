# Audio Participation Analysis

Speaker diarization and classification pipeline for city council meeting audio from rent control cities.

## Repository Structure

```
audio-participation/
├── README.md
├── .gitignore
│
├── src/
│   ├── diarization/              # Speaker diarization
│   │   ├── run_diarization.py    # Main diarization script
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   └── classification/           # MLLM classification
│       └── mllm_labeling_gemini.py
│
└── data/
    └── rent_control_cities_summary.csv
```

## Data

Audio files are stored separately (Dropbox) due to size.

| City | State | Meetings |
|------|-------|----------|
| Brockton | MA | 503 |
| Asbury Park | NJ | 372 |
| Jersey City | NJ | 301 |
| New Brunswick | NJ | 277 |
| Inglewood | CA | 154 |
| Oakland | CA | 9 |
| East Palo Alto | CA | 2 |

**Total: 1,618 meetings from 7 rent control cities (2017-2023)**

## Quick Start

### 1. Setup

```bash
git clone https://github.com/MenglinMileyLiu/audio-participation.git
cd audio-participation

python -m venv venv
source venv/bin/activate

pip install -r src/diarization/requirements.txt
```

### 2. HuggingFace Token

```bash
# Get token: https://huggingface.co/settings/tokens
# Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1

export HF_TOKEN='your_token_here'
```

### 3. Run Diarization

```bash
python src/diarization/run_diarization.py \
    --audio-dir /path/to/audio \
    --output-dir /path/to/output \
    --batch-all
```

See [src/diarization/README.md](src/diarization/README.md) for detailed instructions.

## Requirements

- Python 3.10+
- GPU recommended (CUDA or Apple MPS)
- HuggingFace account
