"""
Script to test MLLM (Gemini) labeling on audio segments
Compares Gemini's labels with human ground truth from Chino Round 2

This script:
1. Loads ground truth labeled segments
2. Uses Gemini to classify audio as "official" or "citizen"
3. Calculates agreement with human labels
4. Generates detailed comparison report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import google.generativeai as genai
import os
import time
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def setup_gemini():
    """Initialize Gemini API"""
    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n❌ Error: GOOGLE_API_KEY environment variable not set!")
        print("Set it with: export GOOGLE_API_KEY='your_key_here'")
        return None

    genai.configure(api_key=api_key)

    # Use Gemini 2.5 Flash (fast, multimodal audio support)
    model = genai.GenerativeModel('gemini-2.5-flash')

    print("✅ Gemini API initialized successfully")
    return model

def create_labeling_prompt():
    """Create the prompt for audio classification"""
    prompt = """You are an expert in analyzing city council meeting audio.

Your task is to classify this audio segment as either "official" or "citizen" based on the speaker's role.

OFFICIAL speakers are:
- Mayor, council members, city manager
- City staff (city attorney, planning director, department heads)
- Speaking in an official capacity
- Usually have formal titles and roles

CITIZEN speakers are:
- Members of the public during public comment periods
- Community residents speaking about local issues
- People without official government roles in the meeting

Listen to the audio carefully and respond with ONLY ONE WORD:
- "official" if the speaker is a government official or staff member
- "citizen" if the speaker is a member of the public

Response (one word only):"""

    return prompt

def label_audio_with_gemini(model, audio_path, max_retries=3):
    """
    Use Gemini to label an audio segment

    Args:
        model: Gemini model instance
        audio_path: Path to audio file
        max_retries: Number of retry attempts for API errors

    Returns:
        tuple: (predicted_label, confidence, response_text)
    """
    prompt = create_labeling_prompt()

    for attempt in range(max_retries):
        try:
            # Upload audio file
            audio_file = genai.upload_file(path=str(audio_path))

            # Wait for file to be processed
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = genai.get_file(audio_file.name)

            if audio_file.state.name == "FAILED":
                raise ValueError(f"Audio file processing failed: {audio_file.state.name}")

            # Generate content
            response = model.generate_content(
                [audio_file, prompt],
                request_options={"timeout": 60}
            )

            # Parse response
            response_text = response.text.strip().lower()

            # Extract label
            if "official" in response_text:
                label = "official"
                # Higher confidence if it's the only word
                confidence = "high" if response_text == "official" else "medium"
            elif "citizen" in response_text:
                label = "citizen"
                confidence = "high" if response_text == "citizen" else "medium"
            else:
                label = "unknown"
                confidence = "low"

            # Clean up uploaded file
            genai.delete_file(audio_file.name)

            return label, confidence, response_text

        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "error", "low", str(e)

    return "error", "low", "Max retries exceeded"

def main():
    print("=" * 80)
    print("MLLM LABELING TEST - Gemini on Chino Ground Truth")
    print("=" * 80)

    # Setup Gemini
    model = setup_gemini()
    if model is None:
        return

    # Load ground truth
    base_dir = Path("output/samples_for_labeling/round2_chino_3meetings_IN_PROGRESS")
    ground_truth_file = base_dir / "GROUND_TRUTH_chino_round2.csv"

    if not ground_truth_file.exists():
        print(f"\n❌ Error: Ground truth file not found: {ground_truth_file}")
        return

    print(f"\nLoading ground truth labels...")
    ground_truth = pd.read_csv(ground_truth_file)
    print(f"  Total ground truth segments: {len(ground_truth)}")

    # Sample 300 segments for testing (stratified by label)
    print("\nSampling 300 segments for testing...")

    # Sample proportionally from each label (maintain ~83% official, 17% citizen ratio)
    n_samples = min(300, len(ground_truth))

    official_samples = ground_truth[ground_truth['final_label'] == 'official'].sample(
        n=min(250, len(ground_truth[ground_truth['final_label'] == 'official'])),
        random_state=42
    )
    citizen_samples = ground_truth[ground_truth['final_label'] == 'citizen'].sample(
        n=min(50, len(ground_truth[ground_truth['final_label'] == 'citizen'])),
        random_state=42
    )

    test_sample = pd.concat([official_samples, citizen_samples]).sample(frac=1, random_state=42)

    print(f"  Test sample size: {len(test_sample)}")
    print(f"    Official: {len(test_sample[test_sample['final_label'] == 'official'])}")
    print(f"    Citizen: {len(test_sample[test_sample['final_label'] == 'citizen'])}")

    # Process each audio segment
    print("\n" + "=" * 80)
    print("LABELING AUDIO SEGMENTS WITH GEMINI")
    print("=" * 80)

    # Check for existing results and resume from where we left off
    temp_file = base_dir / "GEMINI_RESULTS_temp.csv"
    results = []
    processed_segment_ids = set()

    if temp_file.exists():
        print(f"\n✅ Found existing results file - resuming from checkpoint")
        existing_df = pd.read_csv(temp_file)
        results = existing_df.to_dict('records')
        processed_segment_ids = set(existing_df['segment_id'].values)
        print(f"   Already processed: {len(results)} segments")
        print(f"   Remaining: {len(test_sample) - len(results)} segments")
    else:
        print(f"\n▶️  Starting fresh - processing {len(test_sample)} segments")

    for idx, row in test_sample.iterrows():
        segment_id = row['segment_id']

        # Skip if already processed
        if segment_id in processed_segment_ids:
            continue

        audio_file = row['audio_file']
        true_label = row['final_label']

        # Construct full audio path - look in audio_clips folder
        audio_clips_dir = base_dir / "audio_clips"
        audio_path = audio_clips_dir / audio_file

        if not audio_path.exists():
            print(f"\n⚠️  Audio file not found: {audio_path}")
            results.append({
                'segment_id': segment_id,
                'meeting': row['meeting'],
                'speaker_id': row['speaker_id'],
                'true_label': true_label,
                'gemini_label': 'error',
                'gemini_confidence': 'low',
                'gemini_response': 'File not found',
                'agreement': False,
                'audio_file': str(audio_path)
            })
            continue

        print(f"\n[{len(results) + 1}/{len(test_sample)}] Processing {segment_id}...")
        print(f"  Audio: {audio_path.name}")
        print(f"  True label: {true_label}")

        # Label with Gemini
        gemini_label, confidence, response_text = label_audio_with_gemini(model, audio_path)

        print(f"  Gemini label: {gemini_label} (confidence: {confidence})")

        agreement = gemini_label == true_label
        print(f"  Agreement: {'✅ YES' if agreement else '❌ NO'}")

        results.append({
            'segment_id': segment_id,
            'meeting': row['meeting'],
            'speaker_id': row['speaker_id'],
            'true_label': true_label,
            'gemini_label': gemini_label,
            'gemini_confidence': confidence,
            'gemini_response': response_text,
            'agreement': agreement,
            'audio_file': str(audio_path)
        })

        # Save results after EVERY segment to ensure no data loss
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(temp_file, index=False)

        # Print progress every 10 segments
        if len(results) % 10 == 0:
            print(f"\n  💾 Progress saved: {len(results)}/{len(test_sample)} segments ({len(results)/len(test_sample)*100:.1f}%)")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    # Filter out errors
    valid_results = results_df[results_df['gemini_label'].isin(['official', 'citizen'])].copy()

    if len(valid_results) == 0:
        print("\n❌ No valid results to evaluate!")
        return

    print(f"\nValid predictions: {len(valid_results)} / {len(results_df)}")

    # Overall agreement
    overall_agreement = valid_results['agreement'].mean() * 100
    print(f"\nOverall Agreement: {overall_agreement:.1f}%")

    # Cohen's Kappa
    kappa = cohen_kappa_score(valid_results['true_label'], valid_results['gemini_label'])
    print(f"Cohen's Kappa: {kappa:.3f}")

    # Interpretation
    if kappa < 0:
        interpretation = "Poor (less than chance agreement)"
    elif kappa < 0.20:
        interpretation = "Slight"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"

    print(f"  Interpretation: {interpretation}")

    # Confusion matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    cm = confusion_matrix(valid_results['true_label'], valid_results['gemini_label'],
                          labels=['official', 'citizen'])

    print("\n                Predicted")
    print("                Official  Citizen")
    print(f"True Official   {cm[0][0]:>8}  {cm[0][1]:>7}")
    print(f"     Citizen    {cm[1][0]:>8}  {cm[1][1]:>7}")

    # Classification report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print()
    print(classification_report(valid_results['true_label'], valid_results['gemini_label'],
                                target_names=['official', 'citizen']))

    # Agreement by confidence
    print("\n" + "=" * 80)
    print("AGREEMENT BY CONFIDENCE LEVEL")
    print("=" * 80)

    confidence_agreement = valid_results.groupby('gemini_confidence').agg({
        'agreement': ['sum', 'count', 'mean']
    }).round(3)
    confidence_agreement.columns = ['agreements', 'total', 'agreement_rate']
    print()
    print(confidence_agreement)

    # Save results
    output_file = base_dir / "GEMINI_EVALUATION_RESULTS.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved: {output_file}")

    # Save evaluation report
    report_file = base_dir / "GEMINI_EVALUATION_REPORT.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GEMINI MLLM EVALUATION REPORT - Chino Round 2\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Sample Size: {len(results_df)}\n")
        f.write(f"Valid Predictions: {len(valid_results)} ({len(valid_results)/len(results_df)*100:.1f}%)\n\n")
        f.write(f"Overall Agreement: {overall_agreement:.1f}%\n")
        f.write(f"Cohen's Kappa: {kappa:.3f} ({interpretation})\n\n")
        f.write("Confusion Matrix:\n")
        f.write("                Predicted\n")
        f.write("                Official  Citizen\n")
        f.write(f"True Official   {cm[0][0]:>8}  {cm[0][1]:>7}\n")
        f.write(f"     Citizen    {cm[1][0]:>8}  {cm[1][1]:>7}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(valid_results['true_label'], valid_results['gemini_label'],
                                     target_names=['official', 'citizen']))
        f.write("\n\nAgreement by Confidence Level:\n")
        f.write(confidence_agreement.to_string())

    print(f"✅ Evaluation report saved: {report_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if overall_agreement >= 75:
        print(f"\n✅ EXCELLENT! Gemini achieved {overall_agreement:.1f}% agreement with human labels.")
        print("   Recommendation: Deploy Gemini for large-scale labeling!")
    elif overall_agreement >= 60:
        print(f"\n⚠️  MODERATE. Gemini achieved {overall_agreement:.1f}% agreement with human labels.")
        print("   Recommendation: Review disagreements and refine prompt before scaling.")
    else:
        print(f"\n❌ LOW AGREEMENT. Gemini achieved only {overall_agreement:.1f}% agreement.")
        print("   Recommendation: Stick with human labeling or try different MLLM approach.")

    return results_df

if __name__ == "__main__":
    results = main()
