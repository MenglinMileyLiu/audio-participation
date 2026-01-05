"""
Speaker Diarization using pyannote.audio
=========================================

This script performs speaker diarization (identifying who spoke when) using pyannote.audio,
a state-of-the-art speaker diarization system.

What this does:
1. Loads an audio file
2. Identifies different speakers and when they spoke
3. Extracts prosodic features for each speaker segment
4. Saves results with speaker IDs

Requirements:
- pyannote.audio installed
- HuggingFace token with access to pyannote models
  (Get token from: https://huggingface.co/settings/tokens)
  (Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1)

Usage:
    # Single file
    python run_diarization.py --audio-dir /path/to/audio --output-dir /path/to/output

    # Batch processing
    python run_diarization.py --audio-dir /path/to/audio --output-dir /path/to/output --batch file1.mp3 file2.mp3

    # Process all files in a directory
    python run_diarization.py --audio-dir /path/to/audio --output-dir /path/to/output --batch-all
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check for HuggingFace token
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("=" * 80)
    print("ERROR: HuggingFace token not found!")
    print("=" * 80)
    print("\nTo use pyannote.audio, you need a HuggingFace access token.")
    print("\nSteps:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is sufficient)")
    print("3. Accept model conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("4. Set environment variable:")
    print("   export HF_TOKEN='your_token_here'")
    print("\nOr add to your ~/.zshrc or ~/.bashrc:")
    print("   echo 'export HF_TOKEN=\"your_token_here\"' >> ~/.zshrc")
    print("=" * 80)
    sys.exit(1)

try:
    from pyannote.audio import Pipeline
except ImportError:
    print("ERROR: pyannote.audio not installed")
    print("Install with: pip install pyannote.audio")
    sys.exit(1)


def extract_prosodic_features(audio_segment, sr):
    """
    Extract prosodic features from an audio segment.
    Same as before, but now we also have speaker_id.
    """
    features = {}

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if len(pitch_values) > 0:
        features['pitch_mean'] = np.mean(pitch_values)
        features['pitch_std'] = np.std(pitch_values)
        features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_range'] = 0

    # Energy
    rms = librosa.feature.rms(y=audio_segment)[0]
    features['energy_mean'] = np.mean(rms)
    features['energy_std'] = np.std(rms)

    # Zero crossing rate (proxy for speaking rate)
    zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)

    rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])

    # Duration
    features['duration_seconds'] = len(audio_segment) / sr

    return features


def perform_diarization(audio_path, pipeline):
    """
    Perform speaker diarization on an audio file.

    Returns:
        list of dicts with: start, end, speaker_id
    """
    print(f"\nRunning pyannote.audio diarization...")
    print("(This may take a few minutes for long files)")

    # Pre-normalize audio to avoid format issues
    print("  Pre-processing audio to ensure format compatibility...")
    import tempfile
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_audio.name
    temp_audio.close()

    # Load and save with librosa to normalize
    audio_normalized, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    sf.write(temp_path, audio_normalized, sr)
    print(f"  ✓ Audio normalized to {sr} Hz mono")

    # Run diarization on normalized audio
    result = pipeline(temp_path)

    # Clean up temp file
    import os
    os.unlink(temp_path)

    # Extract segments
    segments = []

    # Handle different pyannote API versions
    # In pyannote 3.x, pipeline returns DiarizeOutput with speaker_diarization attribute
    if hasattr(result, 'speaker_diarization'):
        annotation = result.speaker_diarization
    elif hasattr(result, 'annotation'):
        annotation = result.annotation
    else:
        annotation = result

    # Now iterate over the annotation
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker_id': speaker,
            'duration': turn.end - turn.start
        })

    return segments


def process_audio_file(audio_path, output_dir, pipeline):
    """
    Complete processing pipeline:
    1. Load audio
    2. Run speaker diarization
    3. Extract features per segment
    4. Save results
    """
    print("\n" + "=" * 80)
    print(f"Processing: {Path(audio_path).name}")
    print("=" * 80)

    # Check if file exists
    if not Path(audio_path).exists():
        print(f"ERROR: File not found: {audio_path}")
        return None

    # Run diarization
    segments = perform_diarization(audio_path, pipeline)
    print(f"Found {len(segments)} speaker segments")

    # Count unique speakers
    unique_speakers = set([s['speaker_id'] for s in segments])
    print(f"Detected {len(unique_speakers)} unique speakers")

    # Load audio for feature extraction
    print("\nLoading audio for feature extraction...")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration_minutes = len(audio) / sr / 60
    print(f"Audio duration: {duration_minutes:.2f} minutes")

    # Extract features for each segment
    print("\nExtracting prosodic features...")
    segment_features = []

    for i, seg in enumerate(tqdm(segments)):
        # Extract audio segment
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        segment_audio = audio[start_sample:end_sample]

        # Skip very short segments (< 0.5 seconds)
        if len(segment_audio) < sr * 0.5:
            continue

        # Extract features
        features = extract_prosodic_features(segment_audio, sr)
        features['segment_id'] = i
        features['start_time'] = seg['start']
        features['end_time'] = seg['end']
        features['speaker_id'] = seg['speaker_id']
        features['audio_file'] = Path(audio_path).name

        segment_features.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(segment_features)

    # Save results
    output_path = Path(output_dir) / f"{Path(audio_path).stem}_pyannote_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved features to: {output_path}")

    # Save speaker summary
    speaker_summary = []
    for speaker in sorted(unique_speakers):
        speaker_segments = df[df['speaker_id'] == speaker]
        total_time = speaker_segments['duration_seconds'].sum()
        num_segments = len(speaker_segments)
        avg_pitch = speaker_segments['pitch_mean'].mean()
        avg_energy = speaker_segments['energy_mean'].mean()

        speaker_summary.append({
            'speaker_id': speaker,
            'total_speaking_time_seconds': total_time,
            'total_speaking_time_minutes': total_time / 60,
            'num_segments': num_segments,
            'avg_pitch': avg_pitch,
            'avg_energy': avg_energy
        })

    summary_df = pd.DataFrame(speaker_summary)
    summary_path = Path(output_dir) / f"{Path(audio_path).stem}_pyannote_speaker_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved speaker summary to: {summary_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SPEAKER SUMMARY")
    print("=" * 80)
    print(f"{'Speaker':<15} {'Time (min)':<12} {'Segments':<10} {'Avg Pitch':<12} {'Avg Energy'}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['speaker_id']:<15} {row['total_speaking_time_minutes']:>10.2f}  "
              f"{row['num_segments']:>8}  {row['avg_pitch']:>10.1f}  {row['avg_energy']:>10.4f}")

    return df


def is_already_processed(audio_file, output_dir):
    """Check if a file has already been processed"""
    base_name = Path(audio_file).stem
    features_file = output_dir / f"{base_name}_pyannote_features.csv"
    summary_file = output_dir / f"{base_name}_pyannote_speaker_summary.csv"

    if features_file.exists() and summary_file.exists():
        return True
    return False


def batch_process_meetings(meeting_files, audio_dir, output_dir):
    """
    Process multiple meetings with checkpoint/resume capability.
    Skips already-processed files automatically.
    """
    from datetime import datetime

    print("\n" + "=" * 80)
    print("BATCH PROCESSING MODE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Meetings to process: {len(meeting_files)}")

    # Load pipeline once for all files
    print("\nLoading pyannote.audio pipeline...")
    print("(First run will download models - this may take a few minutes)")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"✓ Pipeline loaded on device: {device}")

    except Exception as e:
        print(f"\nERROR loading pipeline: {e}")
        print("\nMake sure you've accepted the model terms at:")
        print("https://huggingface.co/pyannote/speaker-diarization-3.1")
        return

    # Process each meeting
    results = {}
    checkpoint_file = output_dir / "batch_processing_log.txt"

    for i, meeting_file in enumerate(meeting_files, 1):
        print(f"\n\n{'#'*80}")
        print(f"Meeting {i}/{len(meeting_files)}: {meeting_file}")
        print(f"{'#'*80}")

        audio_path = audio_dir / meeting_file

        if not audio_path.exists():
            print(f"  ✗ File not found: {audio_path}")
            results[meeting_file] = "FILE_NOT_FOUND"
            continue

        # Check if already processed
        if is_already_processed(audio_path, output_dir):
            print(f"  ✓ Already processed (skipping)")
            results[meeting_file] = "ALREADY_DONE"

            # Log skip
            with open(checkpoint_file, 'a') as f:
                f.write(f"{datetime.now()}: {meeting_file} - SKIPPED (already processed)\n")
            continue

        # Process meeting
        try:
            df = process_audio_file(audio_path, output_dir, pipeline)

            if df is not None:
                results[meeting_file] = "SUCCESS"
                # Log success
                with open(checkpoint_file, 'a') as f:
                    f.write(f"{datetime.now()}: {meeting_file} - SUCCESS\n")
            else:
                results[meeting_file] = "FAILED"
                with open(checkpoint_file, 'a') as f:
                    f.write(f"{datetime.now()}: {meeting_file} - FAILED\n")

        except Exception as e:
            print(f"\n  ✗ ERROR: {str(e)}")
            results[meeting_file] = "ERROR"
            with open(checkpoint_file, 'a') as f:
                f.write(f"{datetime.now()}: {meeting_file} - ERROR: {str(e)}\n")

    # Final summary
    print("\n\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")

    for meeting, status in results.items():
        if status == "SUCCESS":
            print(f"  ✓ SUCCESS: {meeting}")
        elif status == "ALREADY_DONE":
            print(f"  ✓ SKIPPED: {meeting} (already processed)")
        elif status == "FILE_NOT_FOUND":
            print(f"  ✗ NOT FOUND: {meeting}")
        else:
            print(f"  ✗ FAILED: {meeting}")

    successful = sum(1 for s in results.values() if s == "SUCCESS")
    skipped = sum(1 for s in results.values() if s == "ALREADY_DONE")
    print(f"\nSummary: {successful} processed, {skipped} skipped, {len(results)-successful-skipped} failed")


def main():
    """
    Main execution function.
    Supports both single-file and batch processing modes.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Speaker Diarization using pyannote.audio')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for output files')
    parser.add_argument('--batch', nargs='*', default=None,
                        help='List of specific files to process')
    parser.add_argument('--batch-all', action='store_true',
                        help='Process all mp3/mp4/wav files in audio-dir')
    parser.add_argument('--single', type=str, default=None,
                        help='Process a single file')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PYANNOTE.AUDIO SPEAKER DIARIZATION")
    print("=" * 80)

    # Setup paths from arguments
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")

    # Check audio directory exists
    if not audio_dir.exists():
        print(f"\nERROR: Audio directory not found: {audio_dir}")
        sys.exit(1)

    # Batch all mode - find all audio files
    if args.batch_all:
        meeting_files = []
        for ext in ['*.mp3', '*.mp4', '*.wav', '*.m4a']:
            meeting_files.extend([f.name for f in audio_dir.glob(ext)])
            # Also check subdirectories
            meeting_files.extend([str(f.relative_to(audio_dir)) for f in audio_dir.glob(f'**/{ext}')])
        meeting_files = list(set(meeting_files))  # Remove duplicates
        print(f"\nFound {len(meeting_files)} audio files to process")
        if meeting_files:
            batch_process_meetings(meeting_files, audio_dir, output_dir)
        return

    # Batch mode with specific files
    if args.batch is not None:
        if len(args.batch) == 0:
            print("ERROR: No files specified for batch processing")
            print("Usage: python run_diarization.py --audio-dir /path --output-dir /path --batch file1.mp3 file2.mp3")
            return
        batch_process_meetings(args.batch, audio_dir, output_dir)
        return

    # Single file mode
    print("\nLoading pyannote.audio pipeline...")
    print("(First run will download models - this may take a few minutes)")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"✓ Pipeline loaded on device: {device}")

    except Exception as e:
        print(f"\nERROR loading pipeline: {e}")
        print("\nMake sure you've accepted the model terms at:")
        print("https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)

    # Process single file if specified
    if args.single:
        test_file = audio_dir / args.single
    else:
        # Find first available audio file
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.mp4"))
        if not audio_files:
            print(f"\nERROR: No audio files found in {audio_dir}")
            return
        test_file = audio_files[0]
        print(f"\nNo specific file specified. Using: {test_file.name}")

    if not test_file.exists():
        print(f"\nERROR: File not found: {test_file}")
        return

    df = process_audio_file(test_file, output_dir, pipeline)

    if df is not None:
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Processed {len(df)} segments from {len(df['speaker_id'].unique())} speakers")
        print(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    main()
