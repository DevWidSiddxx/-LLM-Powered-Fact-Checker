import streamlit as st
import spacy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# Page configuration
st.set_page_config(
    page_title="Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .claim-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .verdict-true {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .verdict-false {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .verdict-partial {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .fact-item {
        background-color: #e9ecef;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
if 'embed_model' not in st.session_state:
    st.session_state.embed_model = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'fact_texts' not in st.session_state:
    st.session_state.fact_texts = None

@st.cache_resource
def load_models():
    """Load and cache all models"""
    try:
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Load sentence transformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load classifier
        classifier = pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli")
        
        return nlp, embed_model, classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

@st.cache_data
def load_facts_and_create_index():
    """Load facts from CSV and create FAISS index"""
    try:
        # Check if facts.csv exists
        if not os.path.exists("facts.csv"):
            # Create default facts if file doesn't exist
            default_facts = {
                'id': [1, 2, 3, 4, 5],
                'text': [
                    "The Indian government announced free electricity to farmers in March 2024.",
                    "The Prime Minister launched a crop insurance scheme in 2023.",
                    "Petrol prices were reduced by Rs. 5 in December 2023.",
                    "The government launched a free LPG refill scheme in April 2024.",
                    "No scheme for free laptops to all students has been launched in 2024."
                ]
            }
            df = pd.DataFrame(default_facts)
            df.to_csv("facts.csv", index=False)
        else:
            df = pd.read_csv("facts.csv")
        
        fact_texts = df["text"].tolist()
        
        # Create embeddings using cached model
        if st.session_state.embed_model is not None:
            fact_embeddings = st.session_state.embed_model.encode(fact_texts)
            
            # Create FAISS index
            dimension = fact_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(fact_embeddings))
            
            return fact_texts, index
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error loading facts: {str(e)}")
        return None, None

def extract_claim(text, nlp):
    """Extract claims from input text"""
    doc = nlp(text)
    entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 3]
    
    if not entities or all(ent.lower().isdigit() or ent.lower() in ["true", "false"] for ent in entities):
        return [text.strip()]
    
    return entities

def search_similar_facts(query, embed_model, index, fact_texts, top_k=3):
    """Search for similar facts using FAISS"""
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, dist in zip(indices[0], distances[0]):
        similarity = 1 / (1 + dist)
        results.append((fact_texts[i], similarity))
    
    return results

def get_verdict_improved(claim, evidence_with_scores, embed_model, classifier):
    """Get improved verdict using BART classifier"""
    evidence_texts = [item[0] for item in evidence_with_scores]
    similarities = [item[1] for item in evidence_with_scores]
    
    best_match_idx = similarities.index(max(similarities))
    best_match = evidence_texts[best_match_idx]
    best_similarity = similarities[best_match_idx]
    
    verdict_details = {
        'similarity': best_similarity,
        'best_match': best_match,
        'semantic_similarity': 0,
        'relationship': 'unknown'
    }
    
    if best_similarity > 0.8:
        claim_embedding = embed_model.encode([claim])
        fact_embedding = embed_model.encode([best_match])
        semantic_similarity = cosine_similarity(claim_embedding, fact_embedding)[0][0]
        verdict_details['semantic_similarity'] = semantic_similarity
        
        if semantic_similarity > 0.85:
            return f"✅ TRUE — Claim closely matches verified fact", verdict_details, "true"
        elif semantic_similarity > 0.7:
            return f"🟡 PARTIALLY TRUE — Similar to verified fact but with differences", verdict_details, "partial"
    
    if best_similarity > 0.6:
        try:
            labels = ["contradiction", "entailment", "neutral"]
            result = classifier(f"Claim: {claim}. Fact: {best_match}", labels)
            
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            verdict_details['relationship'] = f"{top_label} ({confidence:.3f})"
            
            if top_label == "contradiction" and confidence > 0.6:
                return f"❌ FALSE — Contradicts verified fact", verdict_details, "false"
            elif top_label == "entailment" and confidence > 0.7:
                return f"✅ TRUE — Supported by verified fact", verdict_details, "true"
        except Exception as e:
            st.warning(f"Classification error: {str(e)}")
    
    # Check for negation patterns
    claim_lower = claim.lower()
    best_match_lower = best_match.lower()
    negation_words = ['no', 'not', 'never', 'none', 'nothing', 'nowhere', 'nobody']
    
    claim_has_negation = any(word in claim_lower for word in negation_words)
    fact_has_negation = any(word in best_match_lower for word in negation_words)
    
    if claim_has_negation != fact_has_negation and best_similarity > 0.6:
        return f"❌ FALSE — Claim contradicts verified fact", verdict_details, "false"
    
    return f"🟡 UNVERIFIABLE — No clear supporting evidence found", verdict_details, "partial"

def get_verdict_simple(claim, evidence_with_scores):
    """Get simple verdict using keyword matching"""
    evidence_texts = [item[0] for item in evidence_with_scores]
    similarities = [item[1] for item in evidence_with_scores]
    
    best_match_idx = similarities.index(max(similarities))
    best_match = evidence_texts[best_match_idx]
    best_similarity = similarities[best_match_idx]
    
    verdict_details = {
        'similarity': best_similarity,
        'best_match': best_match,
        'word_overlap': 0
    }
    
    # Calculate word overlap
    claim_words = set(claim.lower().split())
    fact_words = set(best_match.lower().split())
    common_words = claim_words.intersection(fact_words)
    overlap_ratio = len(common_words) / len(claim_words.union(fact_words))
    verdict_details['word_overlap'] = overlap_ratio
    
    # Check for negation
    negation_words = ['no', 'not', 'never', 'none', 'nothing']
    claim_has_negation = any(word in claim.lower() for word in negation_words)
    fact_has_negation = any(word in best_match.lower() for word in negation_words)
    
    if best_similarity > 0.7 and overlap_ratio > 0.4:
        if claim_has_negation == fact_has_negation:
            return f"✅ TRUE — Matches verified fact", verdict_details, "true"
        else:
            return f"❌ FALSE — Contradicts verified fact", verdict_details, "false"
    elif best_similarity > 0.5 and overlap_ratio > 0.3:
        return f"🟡 PARTIALLY VERIFIABLE — Partially matches", verdict_details, "partial"
    else:
        return f"🟡 UNVERIFIABLE — No clear evidence found", verdict_details, "partial"

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Siddh AI Fact Checker</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading status
        if not st.session_state.models_loaded:
            if st.button("🚀 Load Models", type="primary"):
                with st.spinner("Loading AI models... This may take a few minutes."):
                    nlp, embed_model, classifier = load_models()
                    if nlp and embed_model and classifier:
                        st.session_state.nlp = nlp
                        st.session_state.embed_model = embed_model
                        st.session_state.classifier = classifier
                        
                        # Load facts and create index
                        fact_texts, index = load_facts_and_create_index()
                        if fact_texts and index:
                            st.session_state.fact_texts = fact_texts
                            st.session_state.index = index
                            st.session_state.models_loaded = True
                            st.success("✅ Models loaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load facts database")
                    else:
                        st.error("❌ Failed to load models")
        else:
            st.success("✅ Models Ready")
            
            # Method selection
            st.subheader("🧠 Verification Method")
            method = st.radio(
                "Choose verification approach:",
                ["Advanced (BART Model)", "Simple (Keyword-based)"],
                help="Advanced method uses AI classification, Simple uses keyword matching"
            )
            
            # Facts management
            st.subheader("📚 Facts Database")
            if st.button("📊 View Facts Database"):
                st.session_state.show_facts = True
            
            if st.button("🔄 Reload Facts"):
                st.cache_data.clear()
                fact_texts, index = load_facts_and_create_index()
                if fact_texts and index:
                    st.session_state.fact_texts = fact_texts
                    st.session_state.index = index
                    st.success("Facts reloaded!")
    
    # Main content
    if not st.session_state.models_loaded:
        st.info("👈 Please load the AI models from the sidebar to start fact-checking.")
        
        # Show sample facts while models load
        st.subheader("📚 Sample Facts Database")
        sample_facts = [
            "The Indian government announced free electricity to farmers in March 2024.",
            "The Prime Minister launched a crop insurance scheme in 2023.",
            "Petrol prices were reduced by Rs. 5 in December 2023.",
            "The government launched a free LPG refill scheme in April 2024.",
            "No scheme for free laptops to all students has been launched in 2024."
        ]
        
        for i, fact in enumerate(sample_facts, 1):
            st.markdown(f"**{i}.** {fact}")
            
    else:
        # Show facts database if requested
        if hasattr(st.session_state, 'show_facts') and st.session_state.show_facts:
            st.subheader("📚 Current Facts Database")
            for i, fact in enumerate(st.session_state.fact_texts, 1):
                st.markdown(f"**{i}.** {fact}")
            st.markdown("---")
        
        # Main fact-checking interface
        st.subheader("📝 Enter Statement to Fact-Check")
        
        # Input methods
        input_method = st.radio("Input method:", ["Type", "Upload Text File"], horizontal=True)
        
        user_input = ""
        if input_method == "Type":
            user_input = st.text_area(
                "Enter the statement you want to fact-check:",
                placeholder="e.g., The Indian government has announced free electricity to all farmers starting July 2025.",
                height=100
            )
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file:
                user_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=user_input, height=100, disabled=True)
        
        if st.button("🔍 Fact-Check", type="primary", disabled=not user_input.strip()):
            if user_input.strip():
                with st.spinner("Analyzing statement..."):
                    # Extract claims
                    claims = extract_claim(user_input, st.session_state.nlp)
                    
                    st.subheader("🔍 Analysis Results")
                    
                    for i, claim in enumerate(claims, 1):
                        st.markdown(f'<div class="claim-box"><strong>Claim {i}:</strong> {claim}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Search similar facts
                        similar_facts = search_similar_facts(
                            claim, 
                            st.session_state.embed_model, 
                            st.session_state.index, 
                            st.session_state.fact_texts
                        )
                        
                        # Show similar facts
                        with st.expander(f"📚 Similar Facts (Claim {i})", expanded=False):
                            for j, (fact, similarity) in enumerate(similar_facts, 1):
                                st.markdown(f'<div class="fact-item"><strong>{j}.</strong> {fact}<br>'
                                          f'<small>Similarity: {similarity:.3f}</small></div>', 
                                          unsafe_allow_html=True)
                        
                        # Get verdict
                        use_advanced = method == "Advanced (BART Model)"
                        
                        if use_advanced:
                            verdict, details, verdict_type = get_verdict_improved(
                                claim, similar_facts, 
                                st.session_state.embed_model, 
                                st.session_state.classifier
                            )
                        else:
                            verdict, details, verdict_type = get_verdict_simple(claim, similar_facts)
                        
                        # Display verdict with appropriate styling
                        verdict_class = f"verdict-{verdict_type}"
                        st.markdown(f'<div class="{verdict_class}"><strong>🧠 Verdict:</strong><br>{verdict}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Show technical details
                        with st.expander(f"🔧 Technical Details (Claim {i})", expanded=False):
                            st.json(details)
                        
                        if i < len(claims):
                            st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit | Powered by AI | 🔍 Fact-Checking System"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()