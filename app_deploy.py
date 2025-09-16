import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime

# Lightweight deployment version
st.set_page_config(
    page_title="MedScope AI - Medical Research Finder",
    page_icon="🔬",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster model

@st.cache_data
def load_sample_data():
    # Sample medical papers for demo
    return [
        {
            "title": "Efficacy of Physiotherapy in Knee Osteoarthritis: A Randomized Controlled Trial",
            "abstract": "This randomized controlled trial evaluated the effectiveness of physiotherapy interventions in patients with knee osteoarthritis. Results showed significant improvement in pain scores and functional mobility.",
            "authors": "Smith J, Johnson A, Brown K",
            "year": 2023,
            "journal": "Journal of Orthopedic Medicine"
        },
        {
            "title": "Machine Learning Applications in Medical Diagnosis: A Systematic Review",
            "abstract": "This systematic review examines the current applications of machine learning in medical diagnosis, highlighting recent advances in deep learning for medical imaging and clinical decision support.",
            "authors": "Davis M, Wilson R, Taylor S",
            "year": 2024,
            "journal": "AI in Medicine"
        },
        # Add more sample papers...
    ]

def main():
    st.title("🔬 MedScope AI")
    st.subheader("Medical Research Paper Finder")
    
    # Load model and data
    model = load_model()
    papers = load_sample_data()
    
    # Search interface
    query = st.text_input("Enter your research query:", 
                         placeholder="e.g., latest RCTs on knee osteoarthritis physiotherapy")
    
    if query:
        with st.spinner("Searching medical literature..."):
            # Simple keyword matching for demo (replace with vector search in production)
            results = []
            query_lower = query.lower()
            
            for paper in papers:
                score = 0
                if any(word in paper['abstract'].lower() for word in query_lower.split()):
                    score += 1
                if any(word in paper['title'].lower() for word in query_lower.split()):
                    score += 2
                
                if score > 0:
                    results.append((paper, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            
            if results:
                st.success(f"Found {len(results)} relevant papers")
                
                for i, (paper, score) in enumerate(results[:5]):
                    with st.expander(f"📄 {paper['title']}", expanded=i==0):
                        st.write(f"**Authors:** {paper['authors']}")
                        st.write(f"**Year:** {paper['year']}")
                        st.write(f"**Journal:** {paper['journal']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.write(f"**Relevance Score:** {score}/3")
            else:
                st.warning("No relevant papers found. Try different keywords.")
    
    # Sidebar with info
    with st.sidebar:
        st.header("About MedScope AI")
        st.write("🎯 **Purpose:** Find relevant medical research papers using natural language queries")
        st.write("🔍 **Technology:** Semantic search with sentence transformers")
        st.write("📊 **Dataset:** Curated medical literature abstracts")
        
        st.header("Sample Queries")
        st.write("• Latest RCTs on knee osteoarthritis")
        st.write("• Machine learning in medical diagnosis")
        st.write("• COVID-19 treatment protocols")
        st.write("• Diabetes management strategies")

if __name__ == "__main__":
    main()
