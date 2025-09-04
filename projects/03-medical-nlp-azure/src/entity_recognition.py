"""
Medical Entity Recognition using Azure Cognitive Services and local models
Provides fallback capabilities for offline use.
"""

import pandas as pd
import numpy as np
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import requests
import os
from dataclasses import dataclass

# Local fallback imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    text: str
    category: str
    subcategory: Optional[str] = None
    confidence: float = 0.0
    offset: int = 0
    length: int = 0
    normalized_text: Optional[str] = None

class AzureHealthTextAnalytics:
    """Azure Text Analytics for Health integration."""
    
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Azure Text Analytics client."""
        self.endpoint = endpoint or os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_TEXT_ANALYTICS_KEY')
        self.available = bool(self.endpoint and self.api_key)
        
        if not self.available:
            logging.warning("Azure Text Analytics credentials not found. Using local fallback.")
        
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/json'
        } if self.api_key else {}
    
    def analyze_health_entities(self, text: str) -> List[MedicalEntity]:
        """Analyze text using Azure Text Analytics for Health."""
        if not self.available:
            return []
        
        url = f"{self.endpoint}/text/analytics/v3.1/entities/health/jobs"
        
        payload = {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text
                }
            ]
        }
        
        try:
            # Submit job
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code != 202:
                logging.error(f"Azure API error: {response.status_code}")
                return []
            
            # Get job URL from location header
            job_url = response.headers.get('operation-location')
            if not job_url:
                logging.error("No operation location returned from Azure")
                return []
            
            # Poll for results (simplified for demo)
            import time
            max_attempts = 30
            for attempt in range(max_attempts):
                result_response = requests.get(job_url, headers=self.headers)
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    if result_data.get('status') == 'succeeded':
                        return self._parse_azure_entities(result_data)
                    elif result_data.get('status') in ['failed', 'cancelled']:
                        logging.error(f"Azure job failed: {result_data.get('status')}")
                        return []
                
                time.sleep(2)  # Wait 2 seconds between polls
            
            logging.error("Azure job timed out")
            return []
            
        except Exception as e:
            logging.error(f"Azure API error: {str(e)}")
            return []
    
    def _parse_azure_entities(self, result_data: Dict) -> List[MedicalEntity]:
        """Parse Azure Text Analytics response into MedicalEntity objects."""
        entities = []
        
        try:
            documents = result_data.get('results', {}).get('documents', [])
            if not documents:
                return entities
            
            document = documents[0]  # We only sent one document
            for entity in document.get('entities', []):
                medical_entity = MedicalEntity(
                    text=entity.get('text', ''),
                    category=entity.get('category', ''),
                    subcategory=entity.get('subcategory'),
                    confidence=entity.get('confidenceScore', 0.0),
                    offset=entity.get('offset', 0),
                    length=entity.get('length', 0),
                    normalized_text=entity.get('normalizedText')
                )
                entities.append(medical_entity)
        
        except Exception as e:
            logging.error(f"Error parsing Azure entities: {str(e)}")
        
        return entities

class LocalMedicalNER:
    """Local medical entity recognition using pattern matching and dictionaries."""
    
    def __init__(self):
        """Initialize local NER with medical dictionaries."""
        self.medical_terms = self._load_medical_dictionaries()
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy model not found. Using pattern-based extraction only.")
    
    def _load_medical_dictionaries(self) -> Dict[str, List[str]]:
        """Load medical term dictionaries."""
        return {
            'medications': [
                'aspirin', 'ibuprofen', 'acetaminophen', 'lisinopril', 'metformin',
                'atorvastatin', 'amlodipine', 'omeprazole', 'levothyroxine', 'metoprolol',
                'losartan', 'hydrochlorothiazide', 'gabapentin', 'sertraline', 'prednisone',
                'furosemide', 'warfarin', 'insulin', 'cephalexin', 'albuterol'
            ],
            'conditions': [
                'hypertension', 'diabetes', 'asthma', 'pneumonia', 'bronchitis',
                'arthritis', 'migraine', 'depression', 'anxiety', 'heart failure',
                'atrial fibrillation', 'stroke', 'myocardial infarction', 'angina',
                'copd', 'emphysema', 'cancer', 'tumor', 'infection', 'fracture'
            ],
            'anatomy': [
                'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine',
                'chest', 'abdomen', 'head', 'neck', 'arm', 'leg', 'back', 'spine',
                'joint', 'muscle', 'bone', 'blood', 'vessel', 'artery', 'vein'
            ],
            'procedures': [
                'surgery', 'operation', 'procedure', 'biopsy', 'x-ray', 'ct scan',
                'mri', 'ultrasound', 'ecg', 'ekg', 'blood test', 'urine test',
                'colonoscopy', 'endoscopy', 'angiography', 'mammography'
            ],
            'symptoms': [
                'pain', 'fever', 'nausea', 'vomiting', 'diarrhea', 'constipation',
                'headache', 'dizziness', 'fatigue', 'weakness', 'shortness of breath',
                'chest pain', 'abdominal pain', 'back pain', 'swelling', 'rash'
            ]
        }
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities using local methods."""
        entities: List[MedicalEntity] = []
        if not isinstance(text, str) or not text.strip():
            return entities

        # Expand common medical abbreviations for better matching
        abbr_norm = {
            r"\bdm\b": "diabetes",
            r"\bhtn\b": "hypertension",
            r"\bsob\b": "shortness of breath",
        }
        norm_text = text.lower()
        for pat, repl in abbr_norm.items():
            norm_text = re.sub(pat, repl, norm_text)
        
        # Pattern-based extraction
        singular_map = {
            'medications': 'medication',
            'conditions': 'condition',
            'procedures': 'procedure',
            'symptoms': 'symptoms',
            'anatomy': 'anatomy'
        }

        for category, terms in self.medical_terms.items():
            out_category = singular_map.get(category, category)
            for term in terms:
                # Find all occurrences of the term
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = list(re.finditer(pattern, norm_text))
                
                for match in matches:
                    entity = MedicalEntity(
                        text=term,
                        category=out_category,
                        confidence=0.8,  # Pattern matching confidence
                        offset=match.start(),
                        length=match.end() - match.start()
                    )
                    entities.append(entity)
        
        # spaCy-based extraction if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Map spaCy entity types to medical categories
                category_map = {
                    'PERSON': 'person',
                    'ORG': 'organization',
                    'GPE': 'location',
                    'DATE': 'date',
                    'TIME': 'time',
                    'QUANTITY': 'measurement',
                    'CARDINAL': 'number'
                }
                
                category = category_map.get(ent.label_, 'other')
                entity = MedicalEntity(
                    text=ent.text,
                    category=category,
                    subcategory=ent.label_,
                    confidence=0.9,  # spaCy confidence
                    offset=ent.start_char,
                    length=ent.end_char - ent.start_char
                )
                entities.append(entity)
        
        return entities
    
    def extract_vital_signs(self, text: str) -> List[MedicalEntity]:
        """Extract vital signs using pattern matching."""
        entities = []
        if not isinstance(text, str) or not text.strip():
            return entities
        
        # Blood pressure pattern
        bp_pattern = r'(?:bp|blood pressure):?\s*(\d{2,3})/(\d{2,3})\s*(?:mmhg)?'
        for match in re.finditer(bp_pattern, text.lower()):
            entity = MedicalEntity(
                text=match.group(0),
                category='vital_sign',
                subcategory='blood_pressure',
                confidence=0.95,
                offset=match.start(),
                length=match.end() - match.start(),
                normalized_text=f"{match.group(1)}/{match.group(2)} mmHg"
            )
            entities.append(entity)
        
        # Heart rate pattern
        hr_pattern = r'(?:hr|heart rate):?\s*(\d{2,3})\s*(?:bpm)?'
        for match in re.finditer(hr_pattern, text.lower()):
            entity = MedicalEntity(
                text=match.group(0),
                category='vital_sign',
                subcategory='heart_rate',
                confidence=0.95,
                offset=match.start(),
                length=match.end() - match.start(),
                normalized_text=f"{match.group(1)} bpm"
            )
            entities.append(entity)
        
        # Temperature pattern
        temp_pattern = r'(?:temp|temperature):?\s*(\d{2,3}\.?\d*)\s*(?:f|fahrenheit|c|celsius)?'
        for match in re.finditer(temp_pattern, text.lower()):
            entity = MedicalEntity(
                text=match.group(0),
                category='vital_sign',
                subcategory='temperature',
                confidence=0.95,
                offset=match.start(),
                length=match.end() - match.start(),
                normalized_text=f"{match.group(1)}°F"
            )
            entities.append(entity)
        
        return entities

class MedicalEntityRecognizer:
    """Main class that orchestrates Azure and local entity recognition."""
    
    def __init__(self, azure_endpoint: Optional[str] = None, azure_key: Optional[str] = None):
        """Initialize the medical entity recognizer."""
        self.azure_ner = AzureHealthTextAnalytics(azure_endpoint, azure_key)
        self.local_ner = LocalMedicalNER()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_entities(self, text: str, use_azure: bool = True) -> List[Dict[str, Union[str, float, int]]]:
        """Extract medical entities from text using available methods."""
        all_entities: List[MedicalEntity] = []
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Try Azure first if enabled and available
        if use_azure and self.azure_ner.available:
            try:
                azure_entities = self.azure_ner.analyze_health_entities(text)
                all_entities.extend(azure_entities)
                self.logger.info(f"Azure extracted {len(azure_entities)} entities")
            except Exception as e:
                self.logger.error(f"Azure extraction failed: {str(e)}")
        
        # Always run local extraction for completeness
        local_entities = self.local_ner.extract_entities(text)
        vital_entities = self.local_ner.extract_vital_signs(text)
        
        all_entities.extend(local_entities)
        all_entities.extend(vital_entities)
        
        self.logger.info(f"Local extracted {len(local_entities + vital_entities)} entities")
        
        # Remove duplicates based on text and position
        unique_entities = self._remove_duplicate_entities(all_entities)

        # Normalize abbreviations in entity text (simple map)
        abbr_map = {
            'dm': 'diabetes',
            'htn': 'hypertension',
            'sob': 'shortness of breath'
        }
        for e in unique_entities:
            low = e.text.lower()
            if low in abbr_map:
                e.normalized_text = abbr_map[low]
        
        # Convert to dicts for downstream compatibility and tests
        return [
            {
                'text': e.text,
                'category': e.category,
                'subcategory': e.subcategory,
                'confidence': e.confidence,
                'start': e.offset,
                'end': e.offset + e.length,
                'normalized_text': e.normalized_text,
            }
            for e in unique_entities
        ]
    
    def _remove_duplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities based on overlap and similarity."""
        if not entities:
            return []
        
        # Sort by offset for easier processing
        sorted_entities = sorted(entities, key=lambda x: (x.offset, x.length))
        unique_entities = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps significantly with any existing entity
            is_duplicate = False
            for existing in unique_entities:
                if (abs(entity.offset - existing.offset) < 5 and 
                    entity.text.lower() == existing.text.lower()):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def analyze_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze an entire dataset for medical entities."""
        self.logger.info(f"Starting entity extraction for {len(df)} texts...")
        
        results = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            entities = self.extract_entities(text, use_azure=False)  # Use local for demo
            
            # Aggregate entity information
            entity_summary = {
                'record_id': row.get('record_id', f'record_{idx}'),
                'total_entities': len(entities),
                'entity_categories': list(set([e['category'] for e in entities])),
                'medications_found': [e['text'] for e in entities if e['category'] == 'medication'],
                'conditions_found': [e['text'] for e in entities if e['category'] == 'condition'],
                'vital_signs_found': [e['text'] for e in entities if e['category'] == 'vital_sign'],
                'procedures_found': [e['text'] for e in entities if e['category'] == 'procedure'],
                'all_entities': [{'text': e['text'], 'category': e['category'], 'confidence': e['confidence']} for e in entities]
            }
            
            results.append(entity_summary)
        
        entity_df = pd.DataFrame(results)
        self.logger.info("Entity extraction completed!")
        
        return entity_df
    
    def get_entity_statistics(self, entity_df: pd.DataFrame) -> Dict:
        """Generate statistics about extracted entities."""
        stats = {}
        
        stats['total_records'] = len(entity_df)
        stats['avg_entities_per_record'] = entity_df['total_entities'].mean()
        stats['most_common_categories'] = entity_df['entity_categories'].explode().value_counts().head(10).to_dict()
        
        # Count specific entity types
        stats['records_with_medications'] = (entity_df['medications_found'].str.len() > 0).sum()
        stats['records_with_conditions'] = (entity_df['conditions_found'].str.len() > 0).sum()
        stats['records_with_vital_signs'] = (entity_df['vital_signs_found'].str.len() > 0).sum()
        stats['records_with_procedures'] = (entity_df['procedures_found'].str.len() > 0).sum()
        
        return stats

def main():
    """Test the medical entity recognition."""
    # Test text samples
    test_texts = [
        "Patient presents with chest pain and shortness of breath. BP 140/90 mmHg, HR 85 bpm. Started on lisinopril 10mg and aspirin.",
        "65-year-old male with diabetes and hypertension. Temperature 98.6F. Continue metformin and check blood sugar.",
        "Post-operative day 1 after appendectomy. Patient doing well. Vital signs stable. Continue antibiotics."
    ]
    
    # Initialize recognizer
    recognizer = MedicalEntityRecognizer()
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Test Text {i+1} ---")
        print(f"Text: {text}")
        
        entities = recognizer.extract_entities(text, use_azure=False)
        print(f"\nExtracted {len(entities)} entities:")
        
        for entity in entities:
            print(f"  - {entity.text} ({entity.category}) [confidence: {entity.confidence:.2f}]")

if __name__ == "__main__":
    main()
