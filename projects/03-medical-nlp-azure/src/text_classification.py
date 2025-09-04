"""
Medical Text Classification
Classifies medical documents by type and analyzes sentiment in patient feedback.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import os
import json

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from sklearn.dummy import DummyClassifier

class MedicalTextClassifier:
    """Multi-purpose classifier for medical text classification tasks."""
    
    def __init__(self, task_type: str = 'note_type'):
        """
        Initialize the classifier.
        
        Args:
            task_type: Type of classification task
                      - 'note_type': Classify medical note types
                      - 'sentiment': Analyze sentiment in patient feedback
                      - 'urgency': Classify urgency level
        """
        self.task_type = task_type
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.is_trained = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Task-specific configurations
        self.task_configs = {
            'note_type': {
                'classes': ['clinical_note', 'discharge_summary', 'patient_feedback', 'progress_note'],
                'feature_type': 'tfidf',
                'models': ['logistic_regression', 'random_forest', 'naive_bayes']
            },
            'sentiment': {
                'classes': ['positive', 'negative', 'neutral'],
                'feature_type': 'tfidf',
                'models': ['logistic_regression', 'svm', 'naive_bayes']
            },
            'urgency': {
                'classes': ['low', 'medium', 'high', 'critical'],
                'feature_type': 'tfidf',
                'models': ['logistic_regression', 'random_forest']
            }
        }
    
    def prepare_features(self, texts: List[str], max_features: int = 5000) -> np.ndarray:
        """Prepare text features using TF-IDF or Count vectorization."""
        
        if self.task_type not in self.vectorizers:
            # Initialize vectorizer based on task
            if self.task_configs[self.task_type]['feature_type'] == 'tfidf':
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2),  # Include bigrams
                    min_df=1,  # allow rare terms for tiny demo sets
                    max_df=1.0,
                    lowercase=True,
                    strip_accents='ascii'
                )
            else:
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    lowercase=True,
                    strip_accents='ascii'
                )
            
            # Fit vectorizer on training data
            features = vectorizer.fit_transform(texts)
            self.vectorizers[self.task_type] = vectorizer
        else:
            # Transform using existing vectorizer
            vectorizer = self.vectorizers[self.task_type]
            features = vectorizer.transform(texts)
        
        return features
    
    def train_models(self, df: pd.DataFrame, text_column: str, target_column: str, 
                    test_size: float = 0.2) -> Dict:
        """Train multiple models for the classification task."""
        
        self.logger.info(f"Training models for {self.task_type} classification...")
        
        # Prepare data
        texts = df[text_column].fillna('').astype(str).tolist()
        labels = df[target_column].fillna('unknown').tolist()
        
        # Split data with safeguards for tiny datasets and multi-class stratification
        n_samples = len(texts)
        n_classes = len(set(labels)) if len(labels) else 1
        # Ensure test_size is large enough to include at least one sample per class when stratifying
        min_test_frac = max(n_classes / max(n_samples, 1), test_size)
        adjusted_test_size = min(0.5, max(min_test_frac, test_size))  # cap at 0.5 to keep train sizable

        try:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                texts, labels, test_size=adjusted_test_size, random_state=42, stratify=labels
            )
        except ValueError:
            # Fallback: no stratification
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                texts, labels, test_size=adjusted_test_size, random_state=42, stratify=None
            )
        
        # Prepare features
        X_train = self.prepare_features(X_train_text)
        X_test = self.vectorizers[self.task_type].transform(X_test_text)

        # If the training set has only a single class, use a simple DummyClassifier baseline
        unique_train_classes = list(set(y_train))
        if len(unique_train_classes) < 2:
            self.logger.warning(
                f"Training data for {self.task_type} has a single class ({unique_train_classes[0]}). "
                "Falling back to DummyClassifier for this task."
            )
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)
            try:
                cv_mean = 0.0
                cv_std = 0.0
            except Exception:
                cv_mean = 0.0
                cv_std = 0.0

            results = {
                'dummy_baseline': {
                    'model': dummy,
                    'train_accuracy': dummy.score(X_train, y_train),
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'cv_mean': float(cv_mean),
                    'cv_std': float(cv_std),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
            }
            self.models[self.task_type] = dummy
            self.is_trained = True
            self.logger.info(f"Using DummyClassifier for {self.task_type}")
            return results
        
        # Initialize models
        model_configs = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'svm': SVC(
                kernel='linear', random_state=42, class_weight='balanced', probability=True
            )
        }
        
        results = {}
        
        # Train models specified for this task
        for model_name in self.task_configs[self.task_type]['models']:
            if model_name in model_configs:
                self.logger.info(f"Training {model_name}...")
                
                model = model_configs[model_name]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Cross-validation with safe number of splits
                try:
                    import numpy as np
                    import pandas as pd
                    class_counts = pd.Series(y_train).value_counts()
                    min_class = int(class_counts.min()) if not class_counts.empty else 1
                    max_possible = max(2, min(len(y_train), min_class))
                    cv_splits = max(2, min(5, max_possible))
                    if cv_splits > len(y_train):
                        raise ValueError("Not enough samples for cross-validation")
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splits)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception:
                    cv_scores = None
                    cv_mean = 0.0
                    cv_std = 0.0
                
                # Predictions for detailed evaluation
                y_pred = model.predict(X_test)
                
                results[model_name] = {
                    'model': model,
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': float(cv_mean),
                    'cv_std': float(cv_std),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                self.logger.info(f"{model_name} - Test Accuracy: {test_score:.4f}")
        
        # Store the best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        self.models[self.task_type] = results[best_model_name]['model']
        self.is_trained = True
        
        self.logger.info(f"Best model for {self.task_type}: {best_model_name}")
        
        return results
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Union[str, List[str], np.ndarray]:
        """Make predictions on new texts."""
        
        if not self.is_trained or self.task_type not in self.models:
            raise ValueError(f"Model for {self.task_type} is not trained yet.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_prediction = True
        else:
            single_prediction = False
        
        # Prepare features
        X = self.vectorizers[self.task_type].transform(texts)
        
        # Get model
        model = self.models[self.task_type]
        
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            if single_prediction:
                return probabilities[0]
            return probabilities
        else:
            predictions = model.predict(X)
            if single_prediction:
                return predictions[0]
            return predictions.tolist()
    
    def predict_with_confidence(self, texts: Union[str, List[str]]) -> List[Dict]:
        """Make predictions with confidence scores."""
        
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.predict(texts)
        
        try:
            probabilities = self.predict(texts, return_probabilities=True)
            classes = self.models[self.task_type].classes_
            
            results = []
            for i, pred in enumerate(predictions):
                pred_idx = np.where(classes == pred)[0][0]
                confidence = probabilities[i][pred_idx]
                
                result = {
                    'prediction': pred,
                    'confidence': float(confidence),
                    'all_probabilities': {
                        cls: float(prob) for cls, prob in zip(classes, probabilities[i])
                    }
                }
                results.append(result)
            
            return results
            
        except:
            # Fallback if probabilities not available
            return [{'prediction': pred, 'confidence': 1.0} for pred in predictions]
    
    def analyze_feature_importance(self, top_n: int = 20) -> Dict:
        """Analyze feature importance for the trained model."""
        
        if not self.is_trained or self.task_type not in self.models:
            raise ValueError(f"Model for {self.task_type} is not trained yet.")
        
        model = self.models[self.task_type]
        vectorizer = self.vectorizers[self.task_type]
        feature_names = vectorizer.get_feature_names_out()
        
        importance_analysis = {}
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            
            importance_analysis['top_features'] = [
                {
                    'feature': feature_names[idx],
                    'importance': float(importances[idx])
                }
                for idx in top_indices
            ]
            
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.classes_) == 2:
                # Binary classification
                coefficients = model.coef_[0]
                top_positive_indices = np.argsort(coefficients)[-top_n//2:][::-1]
                top_negative_indices = np.argsort(coefficients)[:top_n//2]
                
                importance_analysis['positive_features'] = [
                    {
                        'feature': feature_names[idx],
                        'coefficient': float(coefficients[idx])
                    }
                    for idx in top_positive_indices
                ]
                
                importance_analysis['negative_features'] = [
                    {
                        'feature': feature_names[idx],
                        'coefficient': float(coefficients[idx])
                    }
                    for idx in top_negative_indices
                ]
            else:
                # Multi-class classification
                for i, class_name in enumerate(model.classes_):
                    coefficients = model.coef_[i]
                    top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
                    
                    importance_analysis[f'class_{class_name}'] = [
                        {
                            'feature': feature_names[idx],
                            'coefficient': float(coefficients[idx])
                        }
                        for idx in top_indices
                    ]
        
        return importance_analysis
    
    def save_model(self, filepath: str):
        """Save the trained model and associated components."""
        
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'task_type': self.task_type,
            'model': self.models.get(self.task_type),
            'vectorizer': self.vectorizers.get(self.task_type),
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.task_type = model_data['task_type']
        self.models[self.task_type] = model_data['model']
        self.vectorizers[self.task_type] = model_data['vectorizer']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")

class SentimentAnalyzer(MedicalTextClassifier):
    """Specialized sentiment analyzer for patient feedback."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        super().__init__(task_type='sentiment')
        
        # Sentiment-specific features
        self.sentiment_lexicon = self._load_sentiment_lexicon()
    
    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Load sentiment lexicon for medical context."""
        # Simple sentiment lexicon for demo
        return {
            # Positive words
            'excellent': 1.0, 'good': 0.8, 'satisfied': 0.7, 'helpful': 0.6,
            'professional': 0.6, 'caring': 0.8, 'thorough': 0.6, 'understanding': 0.7,
            'effective': 0.7, 'improved': 0.8, 'better': 0.6, 'pleased': 0.7,
            
            # Negative words
            'poor': -0.8, 'terrible': -1.0, 'dissatisfied': -0.8, 'rude': -0.9,
            'unprofessional': -0.8, 'rushed': -0.6, 'dismissive': -0.7, 'ineffective': -0.7,
            'worse': -0.8, 'disappointed': -0.7, 'frustrated': -0.6, 'angry': -0.8,
            
            # Neutral/medical terms
            'treatment': 0.0, 'doctor': 0.0, 'nurse': 0.0, 'hospital': 0.0,
            'appointment': 0.0, 'medication': 0.0, 'procedure': 0.0
        }
    
    def extract_sentiment_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentiment-specific features."""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            
            # Lexicon-based features
            positive_score = sum(self.sentiment_lexicon.get(word, 0) for word in words if self.sentiment_lexicon.get(word, 0) > 0)
            negative_score = sum(abs(self.sentiment_lexicon.get(word, 0)) for word in words if self.sentiment_lexicon.get(word, 0) < 0)
            
            # Basic features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            
            # Length features
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            feature_vector = [
                positive_score, negative_score, positive_score - negative_score,
                exclamation_count, question_count, caps_ratio,
                word_count, avg_word_length
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def analyze_sentiment_trends(self, df: pd.DataFrame, text_column: str = 'text', 
                                date_column: str = 'created_date') -> Dict:
        """Analyze sentiment trends over time."""
        
        if not self.is_trained:
            raise ValueError("Sentiment model is not trained yet.")
        
        # Get predictions
        texts = df[text_column].fillna('').astype(str).tolist()
        sentiment_predictions = self.predict_with_confidence(texts)
        
        # Add predictions to dataframe
        results_df = df.copy()
        results_df['predicted_sentiment'] = [pred['prediction'] for pred in sentiment_predictions]
        results_df['sentiment_confidence'] = [pred['confidence'] for pred in sentiment_predictions]
        
        # Analyze trends
        trends = {}
        
        # Overall sentiment distribution
        trends['overall_distribution'] = results_df['predicted_sentiment'].value_counts().to_dict()
        
        # Average confidence by sentiment
        trends['avg_confidence_by_sentiment'] = results_df.groupby('predicted_sentiment')['sentiment_confidence'].mean().to_dict()
        
        # Time-based trends (if date column exists)
        if date_column in results_df.columns:
            try:
                results_df[date_column] = pd.to_datetime(results_df[date_column])
                results_df['date'] = results_df[date_column].dt.date
                
                daily_sentiment = results_df.groupby(['date', 'predicted_sentiment']).size().unstack(fill_value=0)
                trends['daily_sentiment_counts'] = daily_sentiment.to_dict()
                
                # Calculate sentiment score over time (positive=1, neutral=0, negative=-1)
                sentiment_scores = results_df['predicted_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
                daily_scores = results_df.groupby('date').agg({
                    'predicted_sentiment': 'count',
                }).rename(columns={'predicted_sentiment': 'total_feedback'})
                daily_scores['sentiment_score'] = results_df.groupby('date').apply(lambda x: sentiment_scores.loc[x.index].mean())
                
                trends['daily_sentiment_scores'] = daily_scores.to_dict()
                
            except Exception as e:
                self.logger.warning(f"Could not analyze time trends: {str(e)}")
        
        return trends

def main():
    """Test the medical text classification."""
    
    # Create sample data for testing
    sample_data = {
        'text': [
            "Patient presents with chest pain and shortness of breath.",
            "DISCHARGE SUMMARY: Patient discharged home in stable condition.",
            "The staff was excellent and very professional. Great experience!",
            "Day 2: Patient feeling better. Vital signs stable.",
            "Very disappointed with the service. Staff was rude and unprofessional.",
            "Progress note: Continue current medications. Patient improving.",
            "Amazing care received. The nurses were so caring and thorough.",
            "Routine discharge after successful surgery. Follow up in 2 weeks."
        ],
        'note_type': [
            'clinical_note', 'discharge_summary', 'patient_feedback', 'progress_note',
            'patient_feedback', 'progress_note', 'patient_feedback', 'discharge_summary'
        ],
        'sentiment': [
            'neutral', 'neutral', 'positive', 'neutral',
            'negative', 'neutral', 'positive', 'neutral'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test note type classification
    print("=== Testing Note Type Classification ===")
    note_classifier = MedicalTextClassifier(task_type='note_type')
    note_results = note_classifier.train_models(df, 'text', 'note_type')
    
    # Test predictions
    test_text = "Patient reports feeling much better today. Continue current treatment plan."
    prediction = note_classifier.predict(test_text)
    confidence_pred = note_classifier.predict_with_confidence([test_text])
    
    print(f"\nTest text: {test_text}")
    print(f"Predicted note type: {prediction}")
    print(f"Confidence: {confidence_pred[0]}")
    
    # Test sentiment analysis
    print("\n=== Testing Sentiment Analysis ===")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_results = sentiment_analyzer.train_models(df, 'text', 'sentiment')
    
    # Test sentiment prediction
    feedback_text = "The doctor was very professional and helpful. Great experience overall!"
    sentiment_pred = sentiment_analyzer.predict_with_confidence([feedback_text])
    
    print(f"\nFeedback text: {feedback_text}")
    print(f"Predicted sentiment: {sentiment_pred[0]}")
    
    # Feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    try:
        importance = note_classifier.analyze_feature_importance(top_n=10)
        print("Top features for note classification:")
        if 'top_features' in importance:
            for feature in importance['top_features'][:5]:
                print(f"  - {feature['feature']}: {feature['importance']:.4f}")
    except Exception as e:
        print(f"Feature importance analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
