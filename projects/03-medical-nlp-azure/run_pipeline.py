"""
Medical NLP Azure - Main Runner Script
Complete end-to-end medical NLP pipeline with Azure integration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Add shared directory to Python path
shared_dir = current_dir.parent.parent / 'shared'
sys.path.insert(0, str(shared_dir))

from src.nlp_pipeline import MedicalNLPPipeline

# Try importing the shared medical text generator; if unavailable, use a local fallback
try:
    from shared.data_generators.medical_text_generator import MedicalTextGenerator
except Exception as e:
    print(f"Warning: Could not import shared MedicalTextGenerator ({e}). Using local fallback generator.")

    class MedicalTextGenerator:
        """Local fallback generator producing simple synthetic medical texts."""

        def __init__(self, seed: int = 42):
            import random

            random.seed(seed)

        def generate_dataset(self, total_records: int = 100):
            import random
            import pandas as pd

            samples = [
                {
                    "text": "Patient presents with chest pain and shortness of breath. BP 140/90 mmHg, HR 85 bpm.",
                    "note_type": "clinical_note",
                    "sentiment": "neutral",
                },
                {
                    "text": "DISCHARGE SUMMARY: Patient stable, discharged home with medications. Follow up in 2 weeks.",
                    "note_type": "discharge_summary",
                    "sentiment": "positive",
                },
                {
                    "text": "Patient reports severe pain 8/10, requesting stronger pain medication. Appears distressed.",
                    "note_type": "clinical_note",
                    "sentiment": "negative",
                },
                {
                    "text": "PROGRESS NOTE: Patient tolerating treatment well. Vital signs stable. Continue current plan.",
                    "note_type": "progress_note",
                    "sentiment": "positive",
                },
                {
                    "text": "Excellent care from nursing staff. Very satisfied with treatment received. Thank you!",
                    "note_type": "patient_feedback",
                    "sentiment": "positive",
                },
                {
                    "text": "Patient prescribed metformin 500mg twice daily for diabetes management.",
                    "note_type": "clinical_note",
                    "sentiment": "neutral",
                },
                {
                    "text": "Disappointed with wait time. Staff seemed rushed and not attentive to concerns.",
                    "note_type": "patient_feedback",
                    "sentiment": "negative",
                },
                {
                    "text": "Follow-up lab results show improvement in HbA1c from 8.2% to 7.1%. Continue current therapy.",
                    "note_type": "progress_note",
                    "sentiment": "positive",
                },
            ]

            data = []
            for i in range(total_records):
                sample = random.choice(samples)
                data.append(
                    {
                        "text": sample["text"],
                        "note_type": sample["note_type"],
                        "sentiment": sample["sentiment"],
                        "record_id": f"record_{i + 1:04d}",
                    }
                )

            return pd.DataFrame(data)

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data',
        'outputs',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ensured: {directory}")

def generate_synthetic_data(num_records: int = 1000, output_path: str = 'data/medical_text_data.csv'):
    """Generate synthetic medical text data."""
    print(f"Generating {num_records} synthetic medical text records...")
    
    generator = MedicalTextGenerator(seed=42)
    data = generator.generate_dataset(total_records=num_records)
    
    # Save the data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    
    print(f"✓ Synthetic data saved to: {output_path}")
    print(f"✓ Generated {len(data)} records")
    print("✓ Note type distribution:")
    print(data['note_type'].value_counts())
    
    return data

def run_complete_pipeline(data_path: str, config_path: str = None):
    """Run the complete medical NLP pipeline."""
    
    print("\n" + "="*60)
    print("STARTING MEDICAL NLP PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = MedicalNLPPipeline(config_path=config_path)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pipeline.load_data(data_path)
    
    # Preprocess texts for model training
    print("\nPreprocessing texts for model training...")
    processed_df, _ = pipeline.preprocess_texts(df, text_column='text')

    # Train models on preprocessed text (expanded_text)
    print("\nTraining classification models...")
    classification_results = pipeline.train_classification_models(processed_df, text_column='expanded_text')
    
    # Run complete analysis
    print("\nRunning complete NLP analysis...")
    results = pipeline.analyze_complete_dataset(df)
    
    # Save models
    print("\nSaving trained models...")
    pipeline.save_models()
    
    # Save results
    print("\nSaving analysis results...")
    pipeline.save_results()
    
    # Generate and display report
    print("\n" + "="*60)
    print("ANALYSIS REPORT")
    print("="*60)
    report = pipeline.generate_report()
    print(report)
    
    # Demo single text analysis
    print("\n" + "="*60)
    print("SINGLE TEXT ANALYSIS DEMO")
    print("="*60)
    
    demo_texts = [
        "Patient presents with acute chest pain and shortness of breath. Vital signs: BP 150/95 mmHg, HR 92 bpm, Temp 98.6°F. Started on aspirin and nitroglycerin.",
        "The care I received was absolutely excellent! The nursing staff was so professional and caring. Dr. Johnson explained everything clearly and made me feel comfortable.",
        "DISCHARGE SUMMARY: 72-year-old female admitted with pneumonia. Treated with antibiotics and supportive care. Discharged home in stable condition with follow-up scheduled."
    ]
    
    for i, text in enumerate(demo_texts, 1):
        print(f"\n--- Demo Text {i} ---")
        print(f"Text: {text[:100]}...")
        
        result = pipeline.analyze_single_text(text)
        
        print(f"Entities found: {len(result['entities'])}")
        for entity in result['entities'][:3]:  # Show top 3
            print(f"  • {entity['text']} ({entity['category']}) [confidence: {entity['confidence']:.2f}]")
        
        if result['predictions']:
            for pred_type, pred_data in result['predictions'].items():
                print(f"Predicted {pred_type}: {pred_data['prediction']} (confidence: {pred_data['confidence']:.3f})")
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"✓ Results saved to: outputs/")
    print(f"✓ Models saved to: models/")
    
    return pipeline

def run_azure_demo(text: str):
    """Demonstrate Azure Text Analytics integration."""
    
    print("\n" + "="*60)
    print("AZURE TEXT ANALYTICS DEMO")
    print("="*60)
    
    # Check for Azure credentials
    azure_endpoint = os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')
    azure_key = os.getenv('AZURE_TEXT_ANALYTICS_KEY')
    
    if not (azure_endpoint and azure_key):
        print("❌ Azure credentials not found in environment variables.")
        print("   Set AZURE_TEXT_ANALYTICS_ENDPOINT and AZURE_TEXT_ANALYTICS_KEY")
        print("   Using local models instead...")
        return
    
    # Initialize pipeline with Azure
    config = {
        'azure_endpoint': azure_endpoint,
        'azure_key': azure_key,
        'use_azure': True
    }
    
    pipeline = MedicalNLPPipeline()
    pipeline.config.update(config)
    
    print(f"Analyzing text with Azure: {text[:100]}...")
    
    # Analyze with Azure
    result = pipeline.analyze_single_text(text)
    
    print("\nAzure Analysis Results:")
    print(f"Entities found: {len(result['entities'])}")
    for entity in result['entities']:
        print(f"  • {entity['text']} ({entity['category']}) [confidence: {entity['confidence']:.2f}]")
        if entity.get('subcategory'):
            print(f"    Subcategory: {entity['subcategory']}")

def main():
    """Main entry point for the medical NLP pipeline."""
    
    parser = argparse.ArgumentParser(description="Medical NLP Pipeline with Azure Integration")
    parser.add_argument('--mode', choices=['generate', 'train', 'demo', 'azure'], 
                       default='train', help='Mode to run the pipeline')
    parser.add_argument('--data-path', default='data/medical_text_data.csv',
                       help='Path to the medical text data')
    parser.add_argument('--config-path', default='config/default_config.json',
                       help='Path to the configuration file')
    parser.add_argument('--num-records', type=int, default=1000,
                       help='Number of synthetic records to generate')
    parser.add_argument('--demo-text', 
                       default="Patient presents with chest pain and shortness of breath. Blood pressure 140/90.",
                       help='Text for single analysis demo')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    try:
        if args.mode == 'generate':
            # Generate synthetic data only
            generate_synthetic_data(args.num_records, args.data_path)
            
        elif args.mode == 'train':
            # Generate data if it doesn't exist
            if not os.path.exists(args.data_path):
                print("Data file not found. Generating synthetic data...")
                generate_synthetic_data(args.num_records, args.data_path)
            
            # Run complete pipeline
            run_complete_pipeline(args.data_path, args.config_path)
            
        elif args.mode == 'demo':
            # Quick demo with minimal data
            print("Running quick demo with minimal synthetic data...")
            demo_data = generate_synthetic_data(100, 'data/demo_data.csv')
            run_complete_pipeline('data/demo_data.csv', args.config_path)
            
        elif args.mode == 'azure':
            # Azure integration demo
            run_azure_demo(args.demo_text)
        
        print(f"\n🎉 Medical NLP Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
