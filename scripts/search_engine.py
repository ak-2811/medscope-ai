import json
import numpy as np
from typing import List, Dict, Any, Optional
from .embedding_generator import EmbeddingGenerator
import re

class MedScopeSearchEngine:
    """Advanced search engine for medical research papers."""
    
    def __init__(self, embeddings_path: str = "data/embeddings.npy", 
                 metadata_path: str = "data/metadata.json"):
        self.embedding_generator = EmbeddingGenerator()
        
        try:
            self.embedding_generator.load_embeddings(embeddings_path, metadata_path)
            self.is_ready = True
            print("[v0] Search engine initialized successfully")
        except Exception as e:
            print(f"[v0] Error initializing search engine: {e}")
            self.is_ready = False
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess and enhance the search query."""
        # Convert to lowercase for consistency
        query = query.lower().strip()
        
        # Expand common medical abbreviations
        abbreviations = {
            'rct': 'randomized controlled trial',
            'rcts': 'randomized controlled trials',
            'covid': 'covid-19 coronavirus',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'pt': 'physical therapy physiotherapy',
            'oa': 'osteoarthritis',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'mi': 'myocardial infarction',
            'copd': 'chronic obstructive pulmonary disease'
        }
        
        for abbr, expansion in abbreviations.items():
            query = re.sub(rf'\b{abbr}\b', expansion, query)
        
        return query
    
    def filter_by_year(self, results: List[Dict[str, Any]], 
                      year_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter results by publication year."""
        if not year_filter:
            return results
        
        filtered_results = []
        for result in results:
            paper_year = str(result.get('year', ''))
            if year_filter.lower() in ['recent', 'latest']:
                # Consider papers from last 3 years as recent
                if paper_year and int(paper_year) >= 2022:
                    filtered_results.append(result)
            elif year_filter.isdigit():
                if paper_year == year_filter:
                    filtered_results.append(result)
            else:
                filtered_results.append(result)
        
        return filtered_results
    
    def search(self, query: str, top_k: int = 10, 
              year_filter: Optional[str] = None,
              min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Perform semantic search on medical papers."""
        
        if not self.is_ready:
            return [{"error": "Search engine not properly initialized"}]
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        print(f"[v0] Searching for: '{processed_query}'")
        
        # Get similar papers
        results = self.embedding_generator.search_similar_papers(processed_query, top_k * 2)
        
        # Filter by minimum similarity threshold
        results = [r for r in results if r['similarity_score'] >= min_similarity]
        
        # Apply year filter if specified
        if year_filter:
            results = self.filter_by_year(results, year_filter)
        
        # Limit to top_k results
        results = results[:top_k]
        
        # Add search metadata
        for result in results:
            result['search_query'] = query
            result['processed_query'] = processed_query
            result['similarity_percentage'] = round(result['similarity_score'] * 100, 1)
        
        print(f"[v0] Returning {len(results)} results")
        return results
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Generate search suggestions based on partial query."""
        suggestions = []
        
        # Common medical research topics
        topics = [
            "randomized controlled trials",
            "systematic review meta-analysis",
            "covid-19 vaccine effectiveness",
            "machine learning medical diagnosis",
            "physical therapy knee osteoarthritis",
            "telemedicine rural healthcare",
            "mindfulness stress reduction healthcare workers",
            "artificial intelligence radiology",
            "diabetes management interventions",
            "cancer immunotherapy clinical trials"
        ]
        
        partial_lower = partial_query.lower()
        for topic in topics:
            if partial_lower in topic or any(word in topic for word in partial_lower.split()):
                suggestions.append(topic)
        
        return suggestions[:5]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        if not self.is_ready:
            return {"error": "Search engine not initialized"}
        
        metadata = self.embedding_generator.metadata
        
        # Count papers by year
        year_counts = {}
        journal_counts = {}
        
        for paper in metadata:
            year = paper.get('year', 'Unknown')
            journal = paper.get('journal', 'Unknown')
            
            year_counts[year] = year_counts.get(year, 0) + 1
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        
        return {
            "total_papers": len(metadata),
            "papers_by_year": dict(sorted(year_counts.items(), reverse=True)),
            "top_journals": dict(sorted(journal_counts.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]),
            "embedding_dimensions": self.embedding_generator.embeddings.shape[1] if len(self.embedding_generator.embeddings) > 0 else 0
        }

if __name__ == "__main__":
    # Test the search engine
    search_engine = MedScopeSearchEngine()
    
    if search_engine.is_ready:
        # Test queries
        test_queries = [
            "latest RCTs on knee osteoarthritis physiotherapy",
            "machine learning medical diagnosis",
            "COVID-19 vaccine effectiveness",
            "telemedicine rural healthcare"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            results = search_engine.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   Similarity: {result['similarity_percentage']}%")
                print(f"   Authors: {result['authors']}")
                print(f"   Year: {result['year']}")
                print()
    
    print("[v0] Search engine testing complete!")
