"""
Quick smoke test for the Medical NLP Azure pipeline.
Runs a tiny dataset through training and complete analysis, then a single-text analysis.
"""

import pandas as pd
from nlp_pipeline import MedicalNLPPipeline


def main():
    pipeline = MedicalNLPPipeline()

    # Create a small balanced sample (2 per note type; feedback has mixed sentiments)
    sample = pd.DataFrame({
        'record_id': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
        'text': [
            'Patient with HTN and DM. BP 140/90 mmHg, HR 80 bpm.',
            'Clinical note: SOB improved. HR 78 bpm, BP 130/85.',
            'Great care, very professional staff!',
            'Very disappointed, rude staff and long waits.',
            'DISCHARGE SUMMARY: patient discharged home.',
            'Discharge: medications reconciled; follow-up in 1 week.'
        ],
        'note_type': [
            'clinical_note', 'clinical_note',
            'patient_feedback', 'patient_feedback',
            'discharge_summary', 'discharge_summary'
        ],
        'sentiment': [
            'neutral', 'neutral',
            'positive', 'negative',
            'neutral', 'neutral'
        ]
    })

    print('Preprocessing...')
    processed_df, _ = pipeline.preprocess_texts(sample, text_column='text')

    print('Training models...')
    pipeline.train_classification_models(processed_df, text_column='expanded_text')

    print('Running complete analysis...')
    res = pipeline.analyze_complete_dataset(sample, text_column='text')
    print('Processed rows:', len(res['processed_data']))
    print('Entities summary rows:', len(res['entity_data']))

    print('Generating report...')
    print(pipeline.generate_report())

    print('Running single text analysis...')
    single = pipeline.analyze_single_text('Patient c/o SOB and CP. HR 120 bpm, BP 160/95.')
    print('Single tokens (first 5):', single['tokens'][:5])
    print('Single entities count:', len(single['entities']))
    if single['predictions']:
        print('Predictions:', single['predictions'])


if __name__ == '__main__':
    main()
