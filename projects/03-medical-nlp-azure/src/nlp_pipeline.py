"""
Medical NLP Pipeline
Main orchestrator for the complete medical NLP workflow.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import our custom modules
from text_preprocessing import MedicalTextPreprocessor
from entity_recognition import MedicalEntityRecognizer
from text_classification import MedicalTextClassifier, SentimentAnalyzer

class MedicalNLPPipeline:
    """Complete medical NLP pipeline for processing medical texts."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the medical NLP pipeline."""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.preprocessor = MedicalTextPreprocessor(
            preserve_medical_terms=self.config.get('preserve_medical_terms', True)
        )
        
        self.entity_recognizer = MedicalEntityRecognizer(
            azure_endpoint=self.config.get('azure_endpoint'),
            azure_key=self.config.get('azure_key')
        )
        
        self.note_classifier = MedicalTextClassifier(task_type='note_type')
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Pipeline state
        self.is_trained = False
        self.models_trained = {
            'note_classification': False,
            'sentiment_analysis': False
        }
        
        # Results storage
        self.results = {}
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load pipeline configuration."""
        default_config = {
            'preserve_medical_terms': True,
            'use_azure': False,
            'azure_endpoint': None,
            'azure_key': None,
            'output_directory': 'outputs',
            'model_directory': 'models',
            'max_features': 5000,
            'test_size': 0.2,
            'random_state': 42
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Could not load config from {config_path}: {str(e)}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """Load medical text data from file."""
        
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(data_path)
            elif file_type.lower() == 'json':
                df = pd.read_json(data_path)
            elif file_type.lower() == 'excel':
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_texts(self, df: pd.DataFrame, text_column: str = 'text') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess all texts in the dataset."""
        
        self.logger.info("Starting text preprocessing...")
        
        processed_df, features_df = self.preprocessor.preprocess_dataset(df, text_column)
        
        # Store preprocessing summary
        self.results['preprocessing_summary'] = self.preprocessor.get_preprocessing_summary(df, processed_df)
        
        self.logger.info("Text preprocessing completed!")
        return processed_df, features_df
    
    def extract_entities(self, df: pd.DataFrame, text_column: str = 'expanded_text') -> pd.DataFrame:
        """Extract medical entities from all texts."""
        
        self.logger.info("Starting entity extraction...")
        
        entity_df = self.entity_recognizer.analyze_dataset(df, text_column)
        
        # Store entity statistics
        self.results['entity_statistics'] = self.entity_recognizer.get_entity_statistics(entity_df)
        
        self.logger.info("Entity extraction completed!")
        return entity_df
    
    def train_classification_models(self, df: pd.DataFrame, text_column: str = 'expanded_text') -> Dict:
        """Train all classification models."""
        
        self.logger.info("Training classification models...")
        
        classification_results = {}
        
        # Train note type classifier if we have the target column
        if 'note_type' in df.columns:
            self.logger.info("Training note type classifier...")
            note_results = self.note_classifier.train_models(df, text_column, 'note_type')
            classification_results['note_classification'] = note_results
            self.models_trained['note_classification'] = True
        
        # Train sentiment analyzer for patient feedback
        feedback_df = df[df['note_type'] == 'patient_feedback'] if 'note_type' in df.columns else df
        if len(feedback_df) > 0 and 'sentiment' in feedback_df.columns:
            self.logger.info("Training sentiment analyzer...")
            sentiment_results = self.sentiment_analyzer.train_models(feedback_df, text_column, 'sentiment')
            classification_results['sentiment_analysis'] = sentiment_results
            self.models_trained['sentiment_analysis'] = True
        
        self.results['classification_results'] = classification_results
        self.is_trained = any(self.models_trained.values())
        
        self.logger.info("Classification model training completed!")
        return classification_results
    
    def analyze_complete_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> Dict:
        """Run complete analysis on the dataset."""
        
        self.logger.info("Starting complete NLP analysis...")
        
        # Step 1: Preprocessing
        processed_df, features_df = self.preprocess_texts(df, text_column)
        
        # Step 2: Entity extraction
        entity_df = self.extract_entities(processed_df)
        
        # Step 3: Classification (if models are trained)
        if self.is_trained:
            # Note type classification
            if self.models_trained['note_classification']:
                note_predictions = self.note_classifier.predict_with_confidence(
                    processed_df['expanded_text'].tolist()
                )
                processed_df['predicted_note_type'] = [pred['prediction'] for pred in note_predictions]
                processed_df['note_type_confidence'] = [pred['confidence'] for pred in note_predictions]
            
            # Sentiment analysis for feedback
            if 'note_type' in processed_df.columns:
                feedback_mask = processed_df['note_type'] == 'patient_feedback'
            elif 'predicted_note_type' in processed_df.columns:
                feedback_mask = processed_df['predicted_note_type'] == 'patient_feedback'
            else:
                feedback_mask = pd.Series([False] * len(processed_df), index=processed_df.index)

            if feedback_mask.any() and self.models_trained['sentiment_analysis']:
                feedback_texts = processed_df.loc[feedback_mask, 'expanded_text'].tolist()
                sentiment_predictions = self.sentiment_analyzer.predict_with_confidence(feedback_texts)
                
                processed_df.loc[feedback_mask, 'predicted_sentiment'] = [pred['prediction'] for pred in sentiment_predictions]
                processed_df.loc[feedback_mask, 'sentiment_confidence'] = [pred['confidence'] for pred in sentiment_predictions]
        
        # Compile complete results
        complete_results = {
            'preprocessing_summary': self.results.get('preprocessing_summary', {}),
            'entity_statistics': self.results.get('entity_statistics', {}),
            'classification_results': self.results.get('classification_results', {}),
            'processed_data': processed_df,
            'feature_data': features_df,
            'entity_data': entity_df,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.results['complete_analysis'] = complete_results
        
        self.logger.info("Complete NLP analysis finished!")
        return complete_results
    
    def analyze_single_text(self, text: str) -> Dict:
        """Analyze a single text through the complete pipeline."""
        
        # Create temporary dataframe
        temp_df = pd.DataFrame([{'text': text, 'record_id': 'single_analysis'}])
        
        # Preprocessing
        processed_df, _ = self.preprocess_texts(temp_df)
        
        # Entity extraction (prefer expanded text for better matches)
        expanded_text = processed_df['expanded_text'].iloc[0]
        entities = self.entity_recognizer.extract_entities(
            expanded_text,
            use_azure=self.config.get('use_azure', False)
        )
        
        # Classification (if models are trained)
        predictions = {}
        if self.models_trained['note_classification']:
            note_pred = self.note_classifier.predict_with_confidence([processed_df['expanded_text'].iloc[0]])
            predictions['note_type'] = note_pred[0]
        
        if self.models_trained['sentiment_analysis']:
            sentiment_pred = self.sentiment_analyzer.predict_with_confidence([processed_df['expanded_text'].iloc[0]])
            predictions['sentiment'] = sentiment_pred[0]
        
        # Compile results
        analysis_result = {
            'original_text': text,
            'cleaned_text': processed_df['cleaned_text'].iloc[0],
            'expanded_text': processed_df['expanded_text'].iloc[0],
            'tokens': processed_df['tokens'].iloc[0].split(),
            'entities': [
                {
                    'text': e.get('text'),
                    'category': e.get('category'),
                    'subcategory': e.get('subcategory'),
                    'confidence': e.get('confidence', 0.0)
                }
                for e in entities
            ],
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis_result
    
    def save_models(self, model_dir: str = 'models'):
        """Save all trained models."""
        
        if not self.is_trained:
            self.logger.warning("No trained models to save.")
            return
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        if self.models_trained['note_classification']:
            note_model_path = os.path.join(model_dir, 'note_classifier.joblib')
            self.note_classifier.save_model(note_model_path)
        
        if self.models_trained['sentiment_analysis']:
            sentiment_model_path = os.path.join(model_dir, 'sentiment_analyzer.joblib')
            self.sentiment_analyzer.save_model(sentiment_model_path)
        
        # Save pipeline config
        config_path = os.path.join(model_dir, 'pipeline_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = 'models'):
        """Load previously trained models."""
        
        # Load note classifier
        note_model_path = os.path.join(model_dir, 'note_classifier.joblib')
        if os.path.exists(note_model_path):
            self.note_classifier.load_model(note_model_path)
            self.models_trained['note_classification'] = True
        
        # Load sentiment analyzer
        sentiment_model_path = os.path.join(model_dir, 'sentiment_analyzer.joblib')
        if os.path.exists(sentiment_model_path):
            self.sentiment_analyzer.load_model(sentiment_model_path)
            self.models_trained['sentiment_analysis'] = True
        
        self.is_trained = any(self.models_trained.values())
        
        self.logger.info(f"Models loaded from {model_dir}")
    
    def save_results(self, output_dir: str = 'outputs', format: str = 'csv'):
        """Save analysis results to files."""
        
        if 'complete_analysis' not in self.results:
            self.logger.warning("No complete analysis results to save.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.results['complete_analysis']
        
        # Save main datasets
        if format.lower() == 'csv':
            results['processed_data'].to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
            results['feature_data'].to_csv(os.path.join(output_dir, 'feature_data.csv'), index=False)
            results['entity_data'].to_csv(os.path.join(output_dir, 'entity_data.csv'), index=False)
        elif format.lower() == 'json':
            results['processed_data'].to_json(os.path.join(output_dir, 'processed_data.json'), orient='records', indent=2)
            results['feature_data'].to_json(os.path.join(output_dir, 'feature_data.json'), orient='records', indent=2)
            results['entity_data'].to_json(os.path.join(output_dir, 'entity_data.json'), orient='records', indent=2)
        
        # Save summary statistics
        def _to_serializable(obj):
            """Recursively convert objects to JSON-serializable types."""
            try:
                import numpy as np
                import pandas as pd
            except Exception:
                # If numpy/pandas aren't available, fall back to best-effort
                np = None
                pd = None

            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            # Numpy scalar types
            if np is not None and isinstance(obj, (np.integer,)):
                return int(obj)
            if np is not None and isinstance(obj, (np.floating,)):
                return float(obj)
            if np is not None and isinstance(obj, (np.bool_,)):
                return bool(obj)
            # Pandas types
            if pd is not None and isinstance(obj, (pd.Series, pd.Index)):
                return obj.tolist()
            # Datetime-like
            try:
                from datetime import datetime
                if isinstance(obj, datetime):
                    return obj.isoformat()
            except Exception:
                pass
            return obj

        summary = _to_serializable({
            'preprocessing_summary': results['preprocessing_summary'],
            'entity_statistics': results['entity_statistics'],
            'analysis_timestamp': results['analysis_timestamp']
        })

        summary_path = os.path.join(output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_to_serializable)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        
        if 'complete_analysis' not in self.results:
            return "No analysis results available. Please run complete analysis first."
        
        results = self.results['complete_analysis']
        
        report_lines = [
            "="*60,
            "MEDICAL NLP ANALYSIS REPORT",
            "="*60,
            "",
            f"Analysis Timestamp: {results['analysis_timestamp']}",
            "",
            "PREPROCESSING SUMMARY:",
            "-"*30
        ]
        
        # Helper to safely format floats
        def _fmt(val, digits: int = 1):
            try:
                return f"{float(val):.{digits}f}"
            except Exception:
                return str(val)

        # Preprocessing summary
        prep_summary = results['preprocessing_summary']
        report_lines.extend([
            f"Total Records Processed: {prep_summary.get('total_records', 'N/A')}",
            f"Average Original Text Length: {_fmt(prep_summary.get('avg_original_length', 'N/A'))} characters",
            f"Average Cleaned Text Length: {_fmt(prep_summary.get('avg_cleaned_length', 'N/A'))} characters",
            f"Average Token Count: {_fmt(prep_summary.get('avg_token_count', 'N/A'))}",
            ""
        ])
        
        # Entity statistics
        entity_stats = results['entity_statistics']
        report_lines.extend([
            "ENTITY EXTRACTION SUMMARY:",
            "-"*30,
            f"Average Entities per Record: {_fmt(entity_stats.get('avg_entities_per_record', 'N/A'))}",
            f"Records with Medications: {entity_stats.get('records_with_medications', 'N/A')}",
            f"Records with Conditions: {entity_stats.get('records_with_conditions', 'N/A')}",
            f"Records with Vital Signs: {entity_stats.get('records_with_vital_signs', 'N/A')}",
            f"Records with Procedures: {entity_stats.get('records_with_procedures', 'N/A')}",
            ""
        ])
        
        # Classification results
        if 'classification_results' in results:
            class_results = results['classification_results']
            report_lines.extend([
                "CLASSIFICATION RESULTS:",
                "-"*30
            ])
            
            # Note classification
            if 'note_classification' in class_results:
                note_results = class_results['note_classification']
                best_model = max(note_results.keys(), key=lambda x: note_results[x]['test_accuracy'])
                report_lines.extend([
                    f"Note Type Classification - Best Model: {best_model}",
                    f"  Test Accuracy: {note_results[best_model]['test_accuracy']:.4f}",
                    f"  Cross-validation Mean: {note_results[best_model]['cv_mean']:.4f} ± {note_results[best_model]['cv_std']:.4f}",
                    ""
                ])
            
            # Sentiment analysis
            if 'sentiment_analysis' in class_results:
                sentiment_results = class_results['sentiment_analysis']
                best_model = max(sentiment_results.keys(), key=lambda x: sentiment_results[x]['test_accuracy'])
                report_lines.extend([
                    f"Sentiment Analysis - Best Model: {best_model}",
                    f"  Test Accuracy: {sentiment_results[best_model]['test_accuracy']:.4f}",
                    f"  Cross-validation Mean: {sentiment_results[best_model]['cv_mean']:.4f} ± {sentiment_results[best_model]['cv_std']:.4f}",
                    ""
                ])
        
        report_lines.extend([
            "="*60,
            "END OF REPORT",
            "="*60
        ])
        
        return "\n".join(report_lines)

def main():
    """Demonstrate the complete medical NLP pipeline."""
    
    # Initialize pipeline
    pipeline = MedicalNLPPipeline()
    
    # For demo, we'll create some sample data
    print("Creating sample medical text data...")
    
    sample_data = {
        'record_id': [f'REC_{i:03d}' for i in range(1, 11)],
        'text': [
            "Patient presents with chest pain and shortness of breath. BP 140/90 mmHg, HR 85 bpm.",
            "DISCHARGE SUMMARY: 65-year-old male with hypertension and diabetes discharged home.",
            "Excellent care received! The staff was very professional and caring throughout my stay.",
            "Day 3: Patient feeling much better. Temperature normal. Continue current medications.",
            "Very disappointed with the service. Long wait times and unprofessional staff.",
            "Progress note: Vital signs stable. Patient ambulating well. Planning discharge tomorrow.",
            "Outstanding experience! The doctor was thorough and explained everything clearly.",
            "Clinical assessment: Acute bronchitis. Prescribed antibiotics and follow-up in 1 week.",
            "Terrible experience. The nurse was rude and dismissive of my concerns.",
            "Final progress note: Patient ready for discharge. All medications reconciled."
        ],
        'note_type': [
            'clinical_note', 'discharge_summary', 'patient_feedback', 'progress_note',
            'patient_feedback', 'progress_note', 'patient_feedback', 'clinical_note',
            'patient_feedback', 'progress_note'
        ],
        'sentiment': [
            'neutral', 'neutral', 'positive', 'neutral',
            'negative', 'neutral', 'positive', 'neutral',
            'negative', 'neutral'
        ],
        'created_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                        '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Run complete analysis
    print("\nRunning complete NLP analysis...")
    
    # Train models first
    pipeline.train_classification_models(df)
    
    # Run complete analysis
    results = pipeline.analyze_complete_dataset(df)
    
    # Generate and print report
    print("\n" + pipeline.generate_report())
    
    # Demonstrate single text analysis
    print("\n" + "="*60)
    print("SINGLE TEXT ANALYSIS DEMO")
    print("="*60)
    
    test_text = "Patient complaining of severe headache and nausea. BP elevated at 160/95. Started on antihypertensive."
    single_result = pipeline.analyze_single_text(test_text)
    
    print(f"\nOriginal text: {single_result['original_text']}")
    print(f"Cleaned text: {single_result['cleaned_text']}")
    print(f"Entities found: {len(single_result['entities'])}")
    for entity in single_result['entities'][:5]:  # Show first 5
        print(f"  - {entity['text']} ({entity['category']}) [confidence: {entity['confidence']:.2f}]")
    
    if single_result['predictions']:
        print(f"\nPredictions:")
        for pred_type, pred_data in single_result['predictions'].items():
            print(f"  - {pred_type}: {pred_data['prediction']} (confidence: {pred_data['confidence']:.3f})")

if __name__ == "__main__":
    main()
