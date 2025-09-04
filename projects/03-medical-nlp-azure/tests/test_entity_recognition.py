"""
Unit tests for medical entity recognition module.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.entity_recognition import MedicalEntityRecognizer


class TestMedicalEntityRecognizer(unittest.TestCase):
    """Test cases for MedicalEntityRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = MedicalEntityRecognizer()
    
    def test_extract_medications(self):
        """Test medication extraction."""
        text = "Patient prescribed metformin 500mg twice daily and lisinopril 10mg once daily."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Filter for medication entities
        medications = [e for e in entities if e['category'] == 'medication']
        
        self.assertTrue(len(medications) > 0)
        
        # Check for expected medications
        med_texts = [med['text'].lower() for med in medications]
        self.assertTrue(any('metformin' in text for text in med_texts))
        self.assertTrue(any('lisinopril' in text for text in med_texts))
    
    def test_extract_conditions(self):
        """Test medical condition extraction."""
        text = "Patient has history of diabetes mellitus type 2 and hypertension."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Filter for condition entities
        conditions = [e for e in entities if e['category'] == 'condition']
        
        self.assertTrue(len(conditions) > 0)
        
        # Check for expected conditions
        condition_texts = [cond['text'].lower() for cond in conditions]
        self.assertTrue(any('diabetes' in text for text in condition_texts))
        self.assertTrue(any('hypertension' in text for text in condition_texts))
    
    def test_extract_procedures(self):
        """Test medical procedure extraction."""
        text = "Patient underwent colonoscopy and blood work including CBC and CMP."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Filter for procedure entities
        procedures = [e for e in entities if e['category'] == 'procedure']
        
        self.assertTrue(len(procedures) > 0)
        
        # Check for expected procedures
        procedure_texts = [proc['text'].lower() for proc in procedures]
        self.assertTrue(any('colonoscopy' in text for text in procedure_texts))
    
    def test_extract_vital_signs(self):
        """Test vital signs extraction."""
        text = "Vital signs: BP 140/90 mmHg, HR 72 bpm, Temperature 98.6°F, O2 saturation 98%"
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Filter for vital sign entities
        vitals = [e for e in entities if e['category'] == 'vital_sign']
        
        self.assertTrue(len(vitals) > 0)
        
        # Check for expected vital signs
        vital_texts = [vital['text'] for vital in vitals]
        self.assertTrue(any('140/90' in text for text in vital_texts))
        self.assertTrue(any('72' in text for text in vital_texts))
        self.assertTrue(any('98.6' in text for text in vital_texts))
    
    def test_confidence_scores(self):
        """Test that entities have confidence scores."""
        text = "Patient takes aspirin 81mg daily for cardioprotection."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        for entity in entities:
            self.assertIn('confidence', entity)
            self.assertIsInstance(entity['confidence'], (int, float))
            self.assertTrue(0 <= entity['confidence'] <= 1)
    
    def test_entity_normalization(self):
        """Test entity normalization."""
        text = "Patient has DM and HTN."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Should normalize abbreviations to full terms
        entity_texts = [e['text'].lower() for e in entities]
        self.assertTrue(any('diabetes' in text for text in entity_texts))
        self.assertTrue(any('hypertension' in text for text in entity_texts))
    
    def test_overlapping_entities(self):
        """Test handling of overlapping entities."""
        text = "Patient has diabetes mellitus type 2 with diabetic nephropathy."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Should handle overlapping terms appropriately
        self.assertTrue(len(entities) > 0)
        
        # Verify no duplicate spans
        spans = [(e['start'], e['end']) for e in entities]
        self.assertEqual(len(spans), len(set(spans)))
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text."""
        # Test empty string
        entities = self.recognizer.extract_entities("", use_azure=False)
        self.assertEqual(len(entities), 0)
        
        # Test None
        entities = self.recognizer.extract_entities(None, use_azure=False)
        self.assertEqual(len(entities), 0)
        
        # Test whitespace only
        entities = self.recognizer.extract_entities("   ", use_azure=False)
        self.assertEqual(len(entities), 0)
    
    def test_case_insensitive_extraction(self):
        """Test case-insensitive entity extraction."""
        text_lower = "patient has diabetes and takes metformin"
        text_upper = "PATIENT HAS DIABETES AND TAKES METFORMIN"
        text_mixed = "Patient Has Diabetes And Takes Metformin"
        
        entities_lower = self.recognizer.extract_entities(text_lower, use_azure=False)
        entities_upper = self.recognizer.extract_entities(text_upper, use_azure=False)
        entities_mixed = self.recognizer.extract_entities(text_mixed, use_azure=False)
        
        # Should extract entities regardless of case
        self.assertTrue(len(entities_lower) > 0)
        self.assertTrue(len(entities_upper) > 0)
        self.assertTrue(len(entities_mixed) > 0)
    
    def test_complex_medical_text(self):
        """Test extraction from complex medical text."""
        text = """
        CHIEF COMPLAINT: Chest pain and shortness of breath.
        
        HISTORY OF PRESENT ILLNESS: 
        65-year-old male with history of hypertension, diabetes mellitus type 2, 
        and hyperlipidemia presents with acute onset chest pain. Patient takes 
        metformin 1000mg twice daily, lisinopril 20mg daily, and atorvastatin 40mg daily.
        
        VITAL SIGNS: 
        BP 150/95 mmHg, HR 88 bpm, RR 18, Temperature 98.4°F, O2 sat 96% on room air.
        
        ASSESSMENT AND PLAN:
        1. Chest pain - rule out acute coronary syndrome
           - Order EKG, troponins, CXR
           - Continue current medications
        2. Diabetes mellitus type 2 - well controlled
           - Continue metformin
           - A1C due
        """
        
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        # Should extract multiple types of entities
        categories = set(e['category'] for e in entities)
        expected_categories = {'condition', 'medication', 'vital_sign', 'procedure'}
        
        # Should find at least some of the expected categories
        self.assertTrue(len(categories.intersection(expected_categories)) > 0)
        
        # Should extract multiple entities
        self.assertTrue(len(entities) >= 5)
    
    def test_medication_dosage_extraction(self):
        """Test extraction of medications with dosages."""
        text = "Prescribed metformin 500mg twice daily and insulin 20 units subcutaneous before meals."
        entities = self.recognizer.extract_entities(text, use_azure=False)
        
        medications = [e for e in entities if e['category'] == 'medication']
        
        # Should capture medication with dosage information
        med_texts = [med['text'] for med in medications]
        self.assertTrue(any('metformin' in text.lower() for text in med_texts))
        self.assertTrue(any('insulin' in text.lower() for text in med_texts))
    
    def test_azure_fallback_handling(self):
        """Test Azure API fallback to local extraction."""
        text = "Patient has diabetes and takes metformin."
        
        # Test without Azure (should use local extraction)
        entities = self.recognizer.extract_entities(text, use_azure=False)
        self.assertTrue(len(entities) > 0)
        
        # Test with Azure but no credentials (should fallback to local)
        entities_fallback = self.recognizer.extract_entities(text, use_azure=True)
        self.assertTrue(len(entities_fallback) > 0)


if __name__ == '__main__':
    unittest.main()
