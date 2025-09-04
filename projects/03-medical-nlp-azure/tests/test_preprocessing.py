"""
Unit tests for medical text preprocessing module.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.text_preprocessing import MedicalTextPreprocessor


class TestMedicalTextPreprocessor(unittest.TestCase):
    """Test cases for MedicalTextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = MedicalTextPreprocessor()
    
    def test_expand_abbreviations(self):
        """Test medical abbreviation expansion."""
        text = "Patient has elevated BP and HR."
        result = self.preprocessor.expand_abbreviations(text)
        self.assertIn("blood pressure", result.lower())
        self.assertIn("heart rate", result.lower())
    
    def test_clean_medical_text(self):
        """Test medical text cleaning."""
        text = "Patient has BP 120/80 mmHg and HR 72 bpm."
        result = self.preprocessor.clean_medical_text(text)
        
        # Should preserve medical terms and units
        self.assertIn("120/80", result)
        self.assertIn("mmHg", result)
        self.assertIn("bpm", result)
        
        # Should expand abbreviations
        self.assertIn("blood pressure", result.lower())
        self.assertIn("heart rate", result.lower())
    
    def test_extract_vital_signs(self):
        """Test vital signs extraction."""
        text = "Patient vitals: BP 140/90 mmHg, HR 85 bpm, Temp 98.6°F, RR 16, O2 Sat 98%"
        vital_signs = self.preprocessor.extract_vital_signs(text)
        
        self.assertTrue(len(vital_signs) > 0)
        
        # Check for expected vital signs
        vital_types = [vs['type'] for vs in vital_signs]
        self.assertIn('blood_pressure', vital_types)
        self.assertIn('heart_rate', vital_types)
        self.assertIn('temperature', vital_types)
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "PATIENT HAS DIABETES  TYPE II   and hypertension!!! "
        result = self.preprocessor.normalize_text(text)
        
        # Should be lowercase
        self.assertEqual(result, result.lower())
        
        # Should remove extra spaces
        self.assertNotIn("  ", result)
        
        # Should remove extra punctuation
        self.assertNotIn("!!!", result)
    
    def test_tokenize_medical_text(self):
        """Test medical text tokenization."""
        text = "Patient prescribed metformin 500mg twice daily."
        tokens = self.preprocessor.tokenize_medical_text(text)
        
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Should preserve medical terms
        token_text = ' '.join(tokens)
        self.assertIn("metformin", token_text)
        self.assertIn("500mg", token_text)
    
    def test_remove_stopwords_preserve_medical(self):
        """Test stopword removal while preserving medical terms."""
        text = "the patient has a history of diabetes"
        tokens = self.preprocessor.tokenize_medical_text(text)
        filtered_tokens = self.preprocessor.remove_stopwords(tokens)
        
        # Should remove common stopwords
        self.assertNotIn("the", filtered_tokens)
        self.assertNotIn("a", filtered_tokens)
        
        # Should preserve medical terms
        self.assertIn("patient", filtered_tokens)
        self.assertIn("diabetes", filtered_tokens)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "Patient presents with elevated BP 140/90 and SOB. PMH significant for DM type II."
        result = self.preprocessor.preprocess_text(text)
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Should contain expanded abbreviations
        self.assertIn("blood pressure", result.lower())
        self.assertIn("shortness of breath", result.lower())
        self.assertIn("diabetes mellitus", result.lower())
    
    def test_preserve_medical_measurements(self):
        """Test preservation of medical measurements and units."""
        text = "Patient weight: 75kg, height: 180cm, BMI: 23.1"
        result = self.preprocessor.clean_medical_text(text)
        
        # Should preserve measurements
        self.assertIn("75kg", result)
        self.assertIn("180cm", result)
        self.assertIn("23.1", result)
    
    def test_handle_empty_text(self):
        """Test handling of empty or None text."""
        # Test empty string
        result = self.preprocessor.preprocess_text("")
        self.assertEqual(result, "")
        
        # Test None
        result = self.preprocessor.preprocess_text(None)
        self.assertEqual(result, "")
        
        # Test whitespace only
        result = self.preprocessor.preprocess_text("   ")
        self.assertEqual(result, "")
    
    def test_configuration_options(self):
        """Test different configuration options."""
        # Test with preserve_medical_terms disabled
        preprocessor_no_preserve = MedicalTextPreprocessor(preserve_medical_terms=False)
        
        text = "Patient has DM and HTN."
        result_preserve = self.preprocessor.preprocess_text(text)
        result_no_preserve = preprocessor_no_preserve.preprocess_text(text)
        
        # With preservation, should expand abbreviations
        self.assertIn("diabetes mellitus", result_preserve.lower())
        self.assertIn("hypertension", result_preserve.lower())


if __name__ == '__main__':
    unittest.main()
