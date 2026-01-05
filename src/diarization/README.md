# Speaker Diarization Pipeline

Speaker diarization for city council meeting audio using pyannote.audio.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get HuggingFace Token

1. Create account: https://huggingface.co/
2. Get token: https://huggingface.co/settings/tokens
3. Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Set environment variable:

```bash
export HF_TOKEN='your_token_here'
```

### 3. Run Diarization

```bash
# Process all files in a directory
python run_diarization.py \
    --audio-dir /path/to/audio/files \
    --output-dir /path/to/output \
    --batch-all

# Process specific files
python run_diarization.py \
    --audio-dir /path/to/audio \
    --output-dir /path/to/output \
    --batch "file1.mp3" "file2.mp3"

# Process single file
python run_diarization.py \
    --audio-dir /path/to/audio \
    --output-dir /path/to/output \
    --single "meeting.mp3"
```

## Output Files

For each audio file:
- `{filename}_pyannote_features.csv` - Segment-level features (42 prosodic features)
- `{filename}_pyannote_speaker_summary.csv` - Per-speaker summary

## GPU Acceleration

The script auto-detects GPU. Check availability:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Checkpointing

- Automatically skips already-processed files
- Progress logged to `batch_processing_log.txt`
- Safe to interrupt and resume
