import streamlit as st
import json
from pathlib import Path
import sys
import os

# Add scripts directory to path for imports
sys.path.append('scripts')

from scripts.search_engine import MedScopeSearchEngine
from scripts.data_processor import MedicalPaperProcessor
from scripts.embedding_generator import EmbeddingGenerator

# Page configuration
st.set_page_config(
    page_title="MedScope AI - Medical Research Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .similarity-score {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .paper-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1565c0;
        margin-bottom: 0.5rem;
    }
    .paper-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .paper-abstract {
        color: #333;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

def initialize_data():
    """Initialize the data and search engine."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if processed data exists
    processed_papers_path = data_dir / "processed_papers.json"
    embeddings_path = data_dir / "embeddings.npy"
    metadata_path = data_dir / "metadata.json"
    
    if not processed_papers_path.exists():
        st.info("Initializing sample dataset...")
        processor = MedicalPaperProcessor()
        sample_papers = processor.create_sample_dataset()
        processor.save_processed_data(sample_papers, str(processed_papers_path))
    
    if not embeddings_path.exists() or not metadata_path.exists():
        st.info("Generating embeddings... This may take a moment.")
        
        with open(processed_papers_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        generator = EmbeddingGenerator()
        embeddings, metadata = generator.generate_embeddings(papers)
        generator.save_embeddings(embeddings, metadata, str(embeddings_path), str(metadata_path))
    
    return MedScopeSearchEngine(str(embeddings_path), str(metadata_path))

def display_paper_result(paper, rank):
    """Display a single paper result."""
    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-title">
            {rank}. {paper['title']}
        </div>
        <div class="paper-meta">
            <strong>Authors:</strong> {paper['authors']} | 
            <strong>Journal:</strong> {paper['journal']} | 
            <strong>Year:</strong> {paper['year']} |
            <span class="similarity-score">Match: {paper['similarity_percentage']}%</span>
        </div>
        <div class="paper-abstract">
            <strong>Abstract:</strong> {paper['abstract']}
        </div>
        {f'<div class="paper-meta"><strong>Keywords:</strong> {paper["keywords"]}</div>' if paper.get('keywords') else ''}
        {f'<div class="paper-meta"><strong>DOI:</strong> {paper["doi"]}</div>' if paper.get('doi') else ''}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🔬 MedScope AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Medical Research Paper Finder</div>', unsafe_allow_html=True)
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        with st.spinner("Initializing MedScope AI..."):
            st.session_state.search_engine = initialize_data()
    
    search_engine = st.session_state.search_engine
    
    if not search_engine.is_ready:
        st.error("Failed to initialize search engine. Please check the data files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Search Options")
        
        # Search parameters
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=10)
        year_filter = st.selectbox(
            "Filter by year",
            options=["All years", "Recent (2022+)", "2024", "2023", "2022"],
            index=0
        )
        min_similarity = st.slider(
            "Minimum similarity (%)", 
            min_value=0, max_value=100, value=10
        ) / 100
        
        st.header("📊 Database Stats")
        stats = search_engine.get_statistics()
        st.metric("Total Papers", stats.get("total_papers", 0))
        st.metric("Embedding Dimensions", stats.get("embedding_dimensions", 0))
        
        # Top journals
        if "top_journals" in stats:
            st.subheader("Top Journals")
            for journal, count in list(stats["top_journals"].items())[:5]:
                if journal != "Unknown":
                    st.text(f"{journal}: {count}")
    
    # Main search interface
    st.header("🔍 Search Medical Literature")
    
    # Search examples
    with st.expander("💡 Example Searches"):
        examples = [
            "latest RCTs on knee osteoarthritis physiotherapy",
            "machine learning applications in medical diagnosis",
            "COVID-19 vaccine effectiveness against variants",
            "telemedicine adoption in rural healthcare",
            "mindfulness interventions for healthcare worker burnout"
        ]
        
        for example in examples:
            if st.button(f"🔍 {example}", key=f"example_{example}"):
                st.session_state.search_query = example
    
    # Search input
    search_query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get('search_query', ''),
        placeholder="e.g., 'latest RCTs on knee osteoarthritis physiotherapy'",
        help="Use natural language to describe what you're looking for"
    )
    
    # Search suggestions
    if search_query and len(search_query) > 3:
        suggestions = search_engine.get_search_suggestions(search_query)
        if suggestions:
            st.write("💡 **Suggestions:**")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(f"🔍 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.search_query = suggestion
                    st.rerun()
    
    # Perform search
    if search_query:
        with st.spinner("Searching medical literature..."):
            year_filter_value = None if year_filter == "All years" else year_filter.split()[0].lower()
            
            results = search_engine.search(
                search_query,
                top_k=top_k,
                year_filter=year_filter_value,
                min_similarity=min_similarity
            )
        
        # Display results
        if results and not results[0].get("error"):
            st.success(f"Found {len(results)} relevant papers")
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Results Found", len(results))
            with col2:
                avg_similarity = sum(r['similarity_score'] for r in results) / len(results)
                st.metric("Avg. Similarity", f"{avg_similarity*100:.1f}%")
            with col3:
                recent_count = sum(1 for r in results if str(r.get('year', '')) >= '2022')
                st.metric("Recent Papers", recent_count)
            
            st.header("📄 Search Results")
            
            # Display each result
            for i, paper in enumerate(results, 1):
                display_paper_result(paper, i)
            
            # Export results
            if st.button("📥 Export Results as JSON"):
                results_json = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"medscope_search_results_{search_query[:30]}.json",
                    mime="application/json"
                )
        
        elif results and results[0].get("error"):
            st.error(f"Search error: {results[0]['error']}")
        else:
            st.warning("No papers found matching your criteria. Try adjusting your search terms or filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🔬 MedScope AI - Powered by Sentence Transformers and Semantic Search</p>
        <p>Built with Streamlit • Vector embeddings using all-mpnet-base-v2</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
