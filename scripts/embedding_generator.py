import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle
from pathlib import Path

class EmbeddingGenerator:
    """Generate vector embeddings for medical research papers."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        print(f"[v0] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.metadata = []
    
    def create_searchable_text(self, paper: Dict[str, Any]) -> str:
        """Combine title, abstract, and keywords for embedding."""
        components = []
        
        if paper.get('title'):
            components.append(f"Title: {paper['title']}")
        
        if paper.get('abstract'):
            components.append(f"Abstract: {paper['abstract']}")
        
        if paper.get('keywords'):
            components.append(f"Keywords: {paper['keywords']}")
        
        return " ".join(components)
    
    def generate_embeddings(self, papers: List[Dict[str, Any]]) -> tuple:
        """Generate embeddings for all papers."""
        print(f"[v0] Generating embeddings for {len(papers)} papers...")
        
        texts = []
        metadata = []
        
        for paper in papers:
            searchable_text = self.create_searchable_text(paper)
            texts.append(searchable_text)
            
            # Store metadata for each paper
            metadata.append({
                'id': paper.get('id', ''),
                'title': paper.get('title', ''),
                'authors': paper.get('authors', ''),
                'journal': paper.get('journal', ''),
                'year': paper.get('year', ''),
                'doi': paper.get('doi', ''),
                'abstract': paper.get('abstract', '')[:500] + "..." if len(paper.get('abstract', '')) > 500 else paper.get('abstract', ''),
                'keywords': paper.get('keywords', '')
            })
        
        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        print(f"[v0] Generated embeddings with shape: {embeddings.shape}")
        
        self.embeddings = embeddings
        self.metadata = metadata
        
        return embeddings, metadata
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], 
                       embeddings_path: str, metadata_path: str):
        """Save embeddings and metadata to files."""
        
        # Save embeddings as numpy array
        np.save(embeddings_path, embeddings)
        
        # Save metadata as JSON
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[v0] Saved embeddings to {embeddings_path}")
        print(f"[v0] Saved metadata to {metadata_path}")
    
    def load_embeddings(self, embeddings_path: str, metadata_path: str):
        """Load embeddings and metadata from files."""
        self.embeddings = np.load(embeddings_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"[v0] Loaded {len(self.embeddings)} embeddings and {len(self.metadata)} metadata entries")
    
    def search_similar_papers(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for papers similar to the query."""
        if len(self.embeddings) == 0:
            print("[v0] No embeddings loaded. Please generate or load embeddings first.")
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Get top-k most similar papers
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = self.metadata[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)
        
        print(f"[v0] Found {len(results)} similar papers for query: '{query}'")
        return results

if __name__ == "__main__":
    # Load processed papers
    with open("data/processed_papers.json", 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings, metadata = generator.generate_embeddings(papers)
    
    # Save embeddings
    Path("data").mkdir(exist_ok=True)
    generator.save_embeddings(
        embeddings, 
        metadata, 
        "data/embeddings.npy", 
        "data/metadata.json"
    )
    
    print("[v0] Embedding generation complete!")
