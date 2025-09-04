"""
Text Preprocessing Pipeline for Medical NLP
Handles cleaning, normalization, and preparation of medical texts.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import string
import logging
from datetime import datetime

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class MedicalTextPreprocessor:
    """Preprocessing pipeline specifically designed for medical texts."""
    
    def __init__(self, preserve_medical_terms: bool = True):
        """Initialize the preprocessor with medical-specific configurations."""
        self.preserve_medical_terms = preserve_medical_terms
        self.lemmatizer = WordNetLemmatizer()
        
        # Medical-specific stop words to remove
        self.medical_stopwords = {
            'patient', 'patients', 'mr', 'mrs', 'ms', 'dr', 'doctor',
            'hospital', 'clinic', 'visit', 'appointment', 'examination',
            'report', 'reports', 'note', 'notes', 'record', 'records'
        }
        
        # Standard stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Medical abbreviations and their expansions
        self.medical_abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'pmh': 'past medical history',
            'wbc': 'white blood cell',
            'rbc': 'red blood cell',
            'hgb': 'hemoglobin',
            'hct': 'hematocrit',
            'bun': 'blood urea nitrogen',
            'cr': 'creatinine',
            'na': 'sodium',
            'k': 'potassium',
            'cl': 'chloride',
            'co2': 'carbon dioxide',
            'mg': 'milligram',
            'ml': 'milliliter',
            'cc': 'cubic centimeter',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'po': 'by mouth',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'sq': 'subcutaneous',
            'stat': 'immediately',
            'npo': 'nothing by mouth',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'abd': 'abdominal',
            'ext': 'extremity',
            'neuro': 'neurological',
            'psych': 'psychiatric',
            'gi': 'gastrointestinal',
            'gu': 'genitourinary',
            'ent': 'ear nose throat',
            'cvs': 'cardiovascular',
            'resp': 'respiratory'
        }
        
        # Medical entities patterns
        self.medication_patterns = [
            r'\b\w+mycin\b',  # antibiotics
            r'\b\w+cillin\b',  # penicillins
            r'\b\w+pril\b',   # ACE inhibitors
            r'\b\w+sartan\b', # ARBs
            r'\b\w+olol\b',   # beta blockers
            r'\b\w+statin\b', # statins
            r'\b\w+zole\b',   # proton pump inhibitors
        ]
        
        self.condition_patterns = [
            r'\b\w+itis\b',   # inflammatory conditions
            r'\b\w+osis\b',   # pathological conditions
            r'\b\w+pathy\b',  # disease conditions
            r'\b\w+emia\b',   # blood conditions
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning for medical documents."""
        if not isinstance(text, str):
            return ""

        # Handle common medical document artifacts
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines
        text = re.sub(r'\t+', ' ', text)  # Replace tabs
        text = re.sub(r'_+', ' ', text)   # Replace underscores

        # Remove document headers/footers patterns
        text = re.sub(r'PATIENT:|DOCTOR:|DATE:|TIME:', '', text, flags=re.IGNORECASE)

        # Normalize medical measurements (concatenate number+unit for common units)
        text = re.sub(r'(\d+)\s*(mg|ml|cc|mcg|kg|lb|cm|mm|in)', r'\1\2', text)
        # Preserve vital sign units like mmHg and bpm as-is (no replacement)

        # Handle common typos in medical texts
        text = re.sub(r'\bpatinet\b', 'patient', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmedication\b', 'medication', text, flags=re.IGNORECASE)

        return text.strip()

    # Backward/compat convenience alias used by tests
    def clean_medical_text(self, text: str) -> str:
        """Alias to clean_text to preserve public API expected by tests."""
        cleaned = self.clean_text(text)
        return self.expand_abbreviations(cleaned)

    def normalize_text(self, text: str) -> str:
        """Normalize case, collapse whitespace, and trim punctuation (keep medical units)."""
        if not isinstance(text, str) or not text.strip():
            return ""

        # Lowercase
        norm = text.lower()
        # Remove excessive punctuation but keep slashes, hyphens, and unit symbols
        norm = re.sub(r"[^\w\s/°.-]", " ", norm)
        # Collapse repeated spaces
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def tokenize_medical_text(self, text: str) -> List[str]:
        """Tokenize while preserving medical tokens like 500mg and 120/80."""
        if not isinstance(text, str) or not text.strip():
            return []

        # Basic normalization first
        text = self.normalize_text(text)

        # Tokenize
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove standard English stopwords, preserving domain terms."""
        if not isinstance(tokens, list):
            return []

        # Use only standard stopwords to avoid removing words like 'patient'
        std_stops = self.stop_words
        filtered = [t for t in tokens if t.lower() not in std_stops]
        return filtered

    def extract_vital_signs(self, text: str) -> List[Dict[str, str]]:
        """Extract common vital signs and return simple dicts with type and value."""
        if not isinstance(text, str) or not text.strip():
            return []

        out: List[Dict[str, str]] = []
        t = text.lower()

        # Blood pressure (e.g., 140/90)
        for m in re.finditer(r"(\d{2,3})/(\d{2,3})\s*(?:mmhg)?", t):
            out.append({
                'type': 'blood_pressure',
                'value': f"{m.group(1)}/{m.group(2)}"
            })

        # Heart rate (e.g., 72 bpm)
        for m in re.finditer(r"(?:hr|heart\s*rate)\s*:?\s*(\d{2,3})\s*(?:bpm)?", t):
            out.append({
                'type': 'heart_rate',
                'value': f"{m.group(1)} bpm"
            })

        # Temperature (e.g., 98.6°F or 37 C)
        for m in re.finditer(r"(?:temp|temperature)\s*:?\s*(\d{2,3}(?:\.\d+)?)\s*(?:°?f|fahrenheit|°?c|celsius)?", t):
            out.append({
                'type': 'temperature',
                'value': f"{m.group(1)}"
            })

        return out

    def preprocess_text(self, text: Optional[str]) -> str:
        """Full preprocessing pipeline for a single text, returning a normalized string.

        - Cleans artifacts
        - Expands medical abbreviations
        - Tokenizes and removes stopwords
        - Lemmatizes tokens
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        cleaned = self.clean_text(text)
        expanded = self.expand_abbreviations(cleaned)
        # For this helper, return expanded text (keep phrases like 'shortness of breath')
        return expanded
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations to full terms."""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check if word is a medical abbreviation
            clean_word = word.lower().strip('.,!?;:')
            if clean_word in self.medical_abbreviations:
                expanded_words.append(self.medical_abbreviations[clean_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using pattern matching and NER."""
        entities = {
            'medications': [],
            'conditions': [],
            'measurements': [],
            'procedures': []
        }
        
        # Extract medications using patterns
        for pattern in self.medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['medications'].extend(matches)
        
        # Extract conditions using patterns
        for pattern in self.condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['conditions'].extend(matches)
        
        # Extract measurements (value + unit)
        measurements = re.findall(r'(\d+\.?\d*)\s*(mg|ml|cc|mcg|kg|lb|cm|mm|in|bpm|mmhg)', text, re.IGNORECASE)
        for val, unit in measurements:
            entities['measurements'].append(f"{val} {unit}")
        
        # Extract procedures (common medical procedures)
        procedure_keywords = [
            'ecg', 'ekg', 'x-ray', 'ct scan', 'mri', 'ultrasound',
            'biopsy', 'surgery', 'procedure', 'examination', 'test'
        ]
        
        for keyword in procedure_keywords:
            if keyword in text.lower():
                entities['procedures'].append(keyword)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def tokenize_and_normalize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize and normalize text for NLP processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but preserve medical terms
        text = re.sub(r'[^\w\s/-]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            # Combine standard and medical stopwords
            all_stopwords = self.stop_words.union(self.medical_stopwords)
            tokens = [token for token in tokens if token not in all_stopwords]
        
        # Remove very short tokens (but preserve medical abbreviations)
        tokens = [token for token in tokens if len(token) > 2 or token.lower() in self.medical_abbreviations]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def extract_features(self, text: str) -> Dict:
        """Extract comprehensive features from medical text."""
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Medical entities
        entities = self.extract_medical_entities(text)
        features['medication_count'] = len(entities['medications'])
        features['condition_count'] = len(entities['conditions'])
        features['measurement_count'] = len(entities['measurements'])
        features['procedure_count'] = len(entities['procedures'])
        
        # Text complexity indicators
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        
        # Medical-specific indicators
        features['contains_vital_signs'] = bool(re.search(r'\d+/\d+|bpm|temp|bp', text.lower()))
        features['contains_medications'] = bool(any(entities['medications']))
        features['contains_procedures'] = bool(any(entities['procedures']))
        
        return features
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess an entire dataset of medical texts."""
        self.logger.info(f"Starting preprocessing of {len(df)} medical texts...")
        
        processed_df = df.copy()
        features_list = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            # Clean and normalize text
            cleaned_text = self.clean_text(text)
            expanded_text = self.expand_abbreviations(cleaned_text)
            
            # Extract entities
            entities = self.extract_medical_entities(expanded_text)
            
            # Extract features
            features = self.extract_features(expanded_text)
            features['record_id'] = row.get('record_id', f'record_{idx}')
            
            # Tokenize for further processing
            tokens = self.tokenize_and_normalize(expanded_text)
            
            # Update the dataframe
            processed_df.loc[idx, 'cleaned_text'] = cleaned_text
            processed_df.loc[idx, 'expanded_text'] = expanded_text
            processed_df.loc[idx, 'tokens'] = ' '.join(tokens)
            processed_df.loc[idx, 'token_count'] = len(tokens)
            
            # Add entity information
            for entity_type, entity_list in entities.items():
                processed_df.loc[idx, f'{entity_type}_extracted'] = ', '.join(entity_list)
            
            features_list.append(features)
        
        # Create features dataframe
        features_df = pd.DataFrame(features_list)
        
        self.logger.info("Preprocessing completed successfully!")
        
        return processed_df, features_df
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
        """Generate a summary of the preprocessing results."""
        summary = {}
        
        # Basic statistics
        summary['total_records'] = len(processed_df)
        summary['avg_original_length'] = original_df['text'].str.len().mean()
        summary['avg_cleaned_length'] = processed_df['cleaned_text'].str.len().mean()
        summary['avg_token_count'] = processed_df['token_count'].mean()
        
        # Entity extraction statistics
        summary['records_with_medications'] = (processed_df['medications_extracted'].str.len() > 0).sum()
        summary['records_with_conditions'] = (processed_df['conditions_extracted'].str.len() > 0).sum()
        summary['records_with_procedures'] = (processed_df['procedures_extracted'].str.len() > 0).sum()
        
        # Note type distribution
        if 'note_type' in processed_df.columns:
            summary['note_type_distribution'] = processed_df['note_type'].value_counts().to_dict()
        
        return summary

def main():
    """Test the preprocessing pipeline."""
    # Test with sample data
    sample_texts = [
        "Patient presents with SOB and CP. BP 140/90 mmHg, HR 85 bpm. Started on lisinopril 10mg bid.",
        "Progress note: Pt feeling better. Temp 98.6F. Continue current medications. D/C planning tomorrow.",
        "Discharge summary: 65 y/o male with HTN and DM. Underwent EKG and CXR. Prescribed metformin 500mg bid."
    ]
    
    preprocessor = MedicalTextPreprocessor()
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Original: {text}")
        
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned: {cleaned}")
        
        expanded = preprocessor.expand_abbreviations(cleaned)
        print(f"Expanded: {expanded}")
        
        entities = preprocessor.extract_medical_entities(expanded)
        print(f"Entities: {entities}")
        
        tokens = preprocessor.tokenize_and_normalize(expanded)
        print(f"Tokens: {tokens}")
        
        features = preprocessor.extract_features(expanded)
        print(f"Features: {dict(list(features.items())[:5])}...")  # Show first 5 features

if __name__ == "__main__":
    main()
