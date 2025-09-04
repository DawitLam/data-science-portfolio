#!/usr/bin/env python3
"""
Medical NLP Azure - Command Line Interface
Main entry point for running the medical NLP pipeline from command line.
"""

import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nlp_pipeline import MedicalNLPPipeline


def analyze_text(args):
    """Analyze a single text string."""
    pipeline = MedicalNLPPipeline()
    
    # Load models if available
    if os.path.exists('models'):
        try:
            pipeline.load_models('models')
            print("✓ Loaded pre-trained models")
        except Exception as e:
            print(f"⚠ Could not load models: {e}")
    
    result = pipeline.analyze_single_text(args.text)
    
    print(f"\n📄 Analysis Results:")
    print(f"Original text: {result['original_text']}")
    print(f"Cleaned text: {result['cleaned_text']}")
    print(f"Entities found: {len(result['entities'])}")
    
    if result['entities']:
        print("\n🏷️  Entities:")
        for entity in result['entities'][:5]:  # Show first 5
            print(f"  - {entity['text']} ({entity['category']}) [confidence: {entity['confidence']:.2f}]")
        if len(result['entities']) > 5:
            print(f"  ... and {len(result['entities']) - 5} more")
    
    if result['predictions']:
        print("\n🤖 Predictions:")
        for pred_type, pred_data in result['predictions'].items():
            print(f"  {pred_type}: {pred_data['prediction']} (confidence: {pred_data['confidence']:.3f})")


def analyze_file(args):
    """Analyze texts from a CSV file."""
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return
    
    pipeline = MedicalNLPPipeline()
    
    # Load data
    df = pd.read_csv(args.file)
    print(f"📊 Loaded {len(df)} records from {args.file}")
    
    text_column = args.text_column
    if text_column not in df.columns:
        print(f"❌ Column '{text_column}' not found. Available columns: {list(df.columns)}")
        return
    
    # Train models if labels are available
    if 'note_type' in df.columns or 'sentiment' in df.columns:
        print("🎯 Training classification models...")
        pipeline.train_classification_models(df, text_column=text_column)
    
    # Run complete analysis
    print("🔄 Running complete analysis...")
    results = pipeline.analyze_complete_dataset(df, text_column=text_column)
    
    # Generate and print report
    print("\n📋 Analysis Report:")
    print(pipeline.generate_report())
    
    # Save results if requested
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        pipeline.save_results(args.output, format='csv')
        print(f"💾 Results saved to {args.output}")


def train_models(args):
    """Train models on a dataset."""
    if not os.path.exists(args.file):
        print(f"❌ Training file not found: {args.file}")
        return
    
    pipeline = MedicalNLPPipeline()
    
    # Load training data
    df = pd.read_csv(args.file)
    print(f"📚 Loading training data: {len(df)} records")
    
    # Check required columns
    text_column = args.text_column
    if text_column not in df.columns:
        print(f"❌ Text column '{text_column}' not found")
        return
    
    # Train models
    print("🎯 Training classification models...")
    results = pipeline.train_classification_models(df, text_column=text_column)
    
    # Print training summary
    print("\n📈 Training Results:")
    for task, task_results in results.items():
        print(f"\n{task.replace('_', ' ').title()}:")
        best_model = max(task_results.keys(), key=lambda x: task_results[x]['test_accuracy'])
        best_acc = task_results[best_model]['test_accuracy']
        print(f"  Best model: {best_model} (accuracy: {best_acc:.4f})")
    
    # Save models
    model_dir = args.model_dir or 'models'
    os.makedirs(model_dir, exist_ok=True)
    pipeline.save_models(model_dir)
    print(f"💾 Models saved to {model_dir}")


def smoke_test(args):
    """Run a quick smoke test."""
    print("🔥 Running smoke test...")
    
    try:
        # Import and run the smoke test
        from quick_smoke import main as smoke_main
        smoke_main()
        print("✅ Smoke test completed successfully!")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical NLP Azure Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze "Patient with chest pain and SOB. BP 150/95."
  %(prog)s file data.csv --text-column medical_text --output results/
  %(prog)s train data.csv --text-column text --model-dir my_models/
  %(prog)s smoke-test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze single text
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single text')
    analyze_parser.add_argument('text', help='Text to analyze')
    analyze_parser.set_defaults(func=analyze_text)
    
    # Analyze file
    file_parser = subparsers.add_parser('file', help='Analyze texts from CSV file')
    file_parser.add_argument('file', help='CSV file path')
    file_parser.add_argument('--text-column', default='text', help='Name of text column (default: text)')
    file_parser.add_argument('--output', help='Output directory for results')
    file_parser.set_defaults(func=analyze_file)
    
    # Train models
    train_parser = subparsers.add_parser('train', help='Train models on a dataset')
    train_parser.add_argument('file', help='Training CSV file path')
    train_parser.add_argument('--text-column', default='text', help='Name of text column (default: text)')
    train_parser.add_argument('--model-dir', help='Directory to save models (default: models)')
    train_parser.set_defaults(func=train_models)
    
    # Smoke test
    smoke_parser = subparsers.add_parser('smoke-test', help='Run quick validation test')
    smoke_parser.set_defaults(func=smoke_test)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
