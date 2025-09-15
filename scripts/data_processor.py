import pandas as pd
import json
import re
from typing import List, Dict, Any
from pathlib import Path

class MedicalPaperProcessor:
    """Process medical research papers and prepare them for vector embedding."""
    
    def __init__(self):
        self.processed_papers = []
    
    def clean_abstract(self, abstract: str) -> str:
        """Clean and normalize abstract text."""
        if not abstract or pd.isna(abstract):
            return ""
        
        # Remove extra whitespace and newlines
        abstract = re.sub(r'\s+', ' ', abstract.strip())
        
        # Remove common formatting artifacts
        abstract = re.sub(r'\[.*?\]', '', abstract)  # Remove citations
        abstract = re.sub(r'©.*?\.', '', abstract)   # Remove copyright
        
        return abstract
    
    def extract_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from paper."""
        return {
            'title': paper.get('title', '').strip(),
            'authors': paper.get('authors', ''),
            'journal': paper.get('journal', ''),
            'year': paper.get('year', ''),
            'doi': paper.get('doi', ''),
            'pmid': paper.get('pmid', ''),
            'keywords': paper.get('keywords', ''),
            'abstract': self.clean_abstract(paper.get('abstract', ''))
        }
    
    def process_csv_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Process medical papers from CSV file."""
        print(f"[v0] Loading dataset from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"[v0] Loaded {len(df)} papers from dataset")
            
            processed_papers = []
            
            for idx, row in df.iterrows():
                paper_data = {
                    'id': f"paper_{idx}",
                    'title': row.get('title', ''),
                    'abstract': row.get('abstract', ''),
                    'authors': row.get('authors', ''),
                    'journal': row.get('journal', ''),
                    'year': row.get('publish_time', row.get('year', '')),
                    'doi': row.get('doi', ''),
                    'pmid': row.get('pmid', ''),
                    'keywords': row.get('keywords', '')
                }
                
                metadata = self.extract_metadata(paper_data)
                
                # Only include papers with meaningful abstracts
                if len(metadata['abstract']) > 50:
                    processed_papers.append(metadata)
            
            print(f"[v0] Processed {len(processed_papers)} papers with valid abstracts")
            return processed_papers
            
        except Exception as e:
            print(f"[v0] Error processing dataset: {e}")
            return []
    
    def create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample dataset for demonstration."""
        sample_papers = [
            {
                'id': 'sample_1',
                'title': 'Effectiveness of Physical Therapy in Knee Osteoarthritis: A Randomized Controlled Trial',
                'abstract': 'Background: Knee osteoarthritis is a common degenerative joint disease affecting millions worldwide. Physical therapy interventions have shown promise in managing symptoms and improving function. Methods: We conducted a randomized controlled trial with 200 patients diagnosed with knee osteoarthritis. Participants were randomly assigned to receive either structured physical therapy (n=100) or standard care (n=100). The physical therapy group received 12 weeks of supervised exercises including strengthening, flexibility, and balance training. Results: The physical therapy group showed significant improvements in pain scores (p<0.001), functional mobility (p<0.01), and quality of life measures (p<0.05) compared to the control group. Conclusion: Structured physical therapy programs are effective in reducing pain and improving function in patients with knee osteoarthritis.',
                'authors': 'Smith, J.A., Johnson, M.B., Williams, C.D.',
                'journal': 'Journal of Orthopedic Physical Therapy',
                'year': '2024',
                'doi': '10.1016/j.jopt.2024.001',
                'pmid': '12345678',
                'keywords': 'knee osteoarthritis, physical therapy, randomized controlled trial, pain management'
            },
            {
                'id': 'sample_2',
                'title': 'Machine Learning Applications in Medical Diagnosis: A Systematic Review',
                'abstract': 'Objective: To systematically review the current applications of machine learning in medical diagnosis across various specialties. Methods: We searched PubMed, Embase, and IEEE databases for studies published between 2020-2024 using machine learning for diagnostic purposes. Results: 156 studies met inclusion criteria, covering radiology (45%), pathology (25%), cardiology (15%), and other specialties (15%). Deep learning models, particularly convolutional neural networks, showed superior performance in image-based diagnoses with average accuracy of 92.3%. Natural language processing models achieved 87.6% accuracy in clinical text analysis. Conclusion: Machine learning shows significant promise in medical diagnosis, with image-based applications leading in performance metrics.',
                'authors': 'Chen, L., Rodriguez, M.A., Patel, S.K.',
                'journal': 'Artificial Intelligence in Medicine',
                'year': '2024',
                'doi': '10.1016/j.artmed.2024.002',
                'pmid': '12345679',
                'keywords': 'machine learning, medical diagnosis, artificial intelligence, systematic review'
            },
            {
                'id': 'sample_3',
                'title': 'COVID-19 Vaccine Effectiveness Against Omicron Variant: Real-World Evidence',
                'abstract': 'Background: The emergence of SARS-CoV-2 Omicron variant raised concerns about vaccine effectiveness. This study evaluates real-world vaccine effectiveness against Omicron infection and severe outcomes. Methods: Population-based cohort study including 2.5 million individuals in Denmark from December 2021 to March 2022. Vaccine effectiveness was estimated using Cox regression models. Results: Two-dose mRNA vaccine effectiveness against Omicron infection was 55.2% (95% CI: 53.1-57.2%) initially, declining to 23.5% after 90 days. Booster dose restored effectiveness to 75.5%. Against hospitalization, effectiveness remained high at 85.4% even after 90 days. Conclusion: While vaccine effectiveness against Omicron infection wanes over time, protection against severe outcomes remains substantial, supporting booster vaccination strategies.',
                'authors': 'Hansen, C.H., Michlmayr, D., Gubbels, S.M.',
                'journal': 'The Lancet',
                'year': '2024',
                'doi': '10.1016/S0140-6736(24)00001-X',
                'pmid': '12345680',
                'keywords': 'COVID-19, vaccine effectiveness, Omicron variant, real-world evidence'
            },
            {
                'id': 'sample_4',
                'title': 'Telemedicine Adoption in Rural Healthcare: Barriers and Facilitators',
                'abstract': 'Introduction: Telemedicine has potential to address healthcare access challenges in rural areas. This study examines factors influencing telemedicine adoption among rural healthcare providers. Methods: Mixed-methods study including surveys of 450 rural healthcare providers and interviews with 25 key stakeholders across 5 states. Results: Main barriers included inadequate internet infrastructure (78%), lack of technical support (65%), and reimbursement concerns (58%). Facilitators included improved patient access (89%), reduced travel burden (82%), and enhanced specialist consultation (76%). Providers with prior telehealth experience showed 3.2x higher adoption rates. Conclusion: Successful telemedicine implementation in rural areas requires addressing infrastructure limitations and providing comprehensive technical support.',
                'authors': 'Thompson, R.L., Garcia, A.M., Brown, K.J.',
                'journal': 'Journal of Rural Health',
                'year': '2023',
                'doi': '10.1111/jrh.2023.001',
                'pmid': '12345681',
                'keywords': 'telemedicine, rural healthcare, healthcare access, digital health'
            },
            {
                'id': 'sample_5',
                'title': 'Mindfulness-Based Stress Reduction for Healthcare Workers: A Meta-Analysis',
                'abstract': 'Objective: Healthcare workers experience high levels of occupational stress and burnout. This meta-analysis evaluates the effectiveness of mindfulness-based stress reduction (MBSR) interventions for healthcare workers. Methods: Systematic search of databases yielded 28 randomized controlled trials (n=2,847 participants). Random-effects meta-analysis was performed using standardized mean differences. Results: MBSR interventions significantly reduced stress levels (SMD=-0.68, 95% CI: -0.89 to -0.47, p<0.001), burnout scores (SMD=-0.52, 95% CI: -0.71 to -0.33, p<0.001), and anxiety symptoms (SMD=-0.45, 95% CI: -0.63 to -0.27, p<0.001). Effects were maintained at 3-month follow-up. Heterogeneity was moderate (I²=45%). Conclusion: MBSR interventions are effective in reducing stress and burnout among healthcare workers, supporting their implementation in healthcare settings.',
                'authors': 'Kumar, S., Lee, H.J., Anderson, P.M.',
                'journal': 'Journal of Occupational Health Psychology',
                'year': '2023',
                'doi': '10.1037/ocp0000345',
                'pmid': '12345682',
                'keywords': 'mindfulness, stress reduction, healthcare workers, burnout, meta-analysis'
            }
        ]
        
        processed_papers = []
        for paper in sample_papers:
            metadata = self.extract_metadata(paper)
            processed_papers.append(metadata)
        
        print(f"[v0] Created sample dataset with {len(processed_papers)} papers")
        return processed_papers
    
    def save_processed_data(self, papers: List[Dict[str, Any]], output_path: str):
        """Save processed papers to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"[v0] Saved {len(papers)} processed papers to {output_path}")

if __name__ == "__main__":
    processor = MedicalPaperProcessor()
    
    # Create sample dataset for demonstration
    sample_papers = processor.create_sample_dataset()
    processor.save_processed_data(sample_papers, "data/processed_papers.json")
    
    print("[v0] Data processing pipeline setup complete!")
