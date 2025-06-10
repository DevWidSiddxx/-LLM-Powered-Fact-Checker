import spacy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------
# 1. Load NLP model (for claims)
# -------------------------------
nlp = spacy.load("en_core_web_sm")

def extract_claim(text):
    doc = nlp(text)
    # Get meaningful entities
    entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 3]
    
    # Check for vague/short-only entities
    if not entities or all(ent.lower().isdigit() or ent.lower() in ["true", "false"] for ent in entities):
        return [text.strip()]  # fallback to full input
    
    return entities

# -----------------------------------
# 2. Load facts and create FAISS index
# -----------------------------------
df = pd.read_csv("facts.csv")
fact_texts = df["text"].tolist()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
fact_embeddings = embed_model.encode(fact_texts)

dimension = fact_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(fact_embeddings))

def search_similar_facts(query, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    # Return facts with similarity scores
    results = []
    for i, dist in zip(indices[0], distances[0]):
        # Convert L2 distance to similarity (lower distance = higher similarity)
        similarity = 1 / (1 + dist)
        results.append((fact_texts[i], similarity))
    
    return results

# --------------------------------------
# 3. Load better model for classification
# --------------------------------------
print("⏳ Loading classification model...")
# Using a better model for text classification
classifier = pipeline("zero-shot-classification", 
                     model="facebook/bart-large-mnli")
print("✅ Model loaded!")

# -------------------------------
# 4. Improved verdict function
# -------------------------------
def get_verdict_improved(claim, evidence_with_scores):
    # Extract just the evidence texts and scores
    evidence_texts = [item[0] for item in evidence_with_scores]
    similarities = [item[1] for item in evidence_with_scores]
    
    # Find the most similar fact
    best_match_idx = similarities.index(max(similarities))
    best_match = evidence_texts[best_match_idx]
    best_similarity = similarities[best_match_idx]
    
    print(f"🎯 Best match similarity: {best_similarity:.3f}")
    
    # High similarity threshold for direct matches
    if best_similarity > 0.8:
        # Use semantic similarity to determine if claim matches fact
        claim_embedding = embed_model.encode([claim])
        fact_embedding = embed_model.encode([best_match])
        
        semantic_similarity = cosine_similarity(claim_embedding, fact_embedding)[0][0]
        print(f"🧠 Semantic similarity: {semantic_similarity:.3f}")
        
        if semantic_similarity > 0.85:
            return f"✅ TRUE — Claim closely matches verified fact: '{best_match}'"
        elif semantic_similarity > 0.7:
            return f"🟡 PARTIALLY TRUE — Similar to verified fact but with differences: '{best_match}'"
    
    # Check for contradictions using zero-shot classification
    if best_similarity > 0.6:
        # Create contradiction prompt
        labels = ["contradiction", "entailment", "neutral"]
        result = classifier(f"Claim: {claim}. Fact: {best_match}", labels)
        
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        
        print(f"🔍 Relationship: {top_label} (confidence: {confidence:.3f})")
        
        if top_label == "contradiction" and confidence > 0.6:
            return f"❌ FALSE — Contradicts verified fact: '{best_match}'"
        elif top_label == "entailment" and confidence > 0.7:
            return f"✅ TRUE — Supported by verified fact: '{best_match}'"
    
    # Check for explicit negation patterns
    claim_lower = claim.lower()
    best_match_lower = best_match.lower()
    
    # Look for negation words in claim
    negation_words = ['no', 'not', 'never', 'none', 'nothing', 'nowhere', 'nobody']
    claim_has_negation = any(word in claim_lower for word in negation_words)
    fact_has_negation = any(word in best_match_lower for word in negation_words)
    
    # If one has negation and the other doesn't, and they're similar, it's likely a contradiction
    if claim_has_negation != fact_has_negation and best_similarity > 0.6:
        return f"❌ FALSE — Claim contradicts verified fact: '{best_match}'"
    
    # Default to unverifiable if no clear match
    return f"🟡 UNVERIFIABLE — No clear supporting evidence found. Closest match: '{best_match}' (similarity: {best_similarity:.3f})"

# Alternative simpler verdict function
def get_verdict_simple(claim, evidence_with_scores):
    evidence_texts = [item[0] for item in evidence_with_scores]
    similarities = [item[1] for item in evidence_with_scores]
    
    best_match_idx = similarities.index(max(similarities))
    best_match = evidence_texts[best_match_idx]
    best_similarity = similarities[best_match_idx]
    
    print(f"🎯 Best match similarity: {best_similarity:.3f}")
    
    # Simple keyword-based approach
    claim_words = set(claim.lower().split())
    fact_words = set(best_match.lower().split())
    
    # Calculate word overlap
    common_words = claim_words.intersection(fact_words)
    overlap_ratio = len(common_words) / len(claim_words.union(fact_words))
    
    print(f"📊 Word overlap ratio: {overlap_ratio:.3f}")
    
    # Check for negation
    negation_words = ['no', 'not', 'never', 'none', 'nothing']
    claim_has_negation = any(word in claim.lower() for word in negation_words)
    fact_has_negation = any(word in best_match.lower() for word in negation_words)
    
    if best_similarity > 0.7 and overlap_ratio > 0.4:
        if claim_has_negation == fact_has_negation:
            return f"✅ TRUE — Matches verified fact: '{best_match}'"
        else:
            return f"❌ FALSE — Contradicts verified fact: '{best_match}'"
    elif best_similarity > 0.5 and overlap_ratio > 0.3:
        return f"🟡 PARTIALLY VERIFIABLE — Partially matches: '{best_match}'"
    else:
        return f"🟡 UNVERIFIABLE — No clear evidence found. Closest: '{best_match}'"

# -------------------------------
# 5. Run the full pipeline
# -------------------------------
if __name__ == "__main__":
    print("🚀 Fact-Checking System Ready!")
    print("Choose verdict method:")
    print("1. Advanced (uses BART model)")
    print("2. Simple (keyword-based)")
    
    method = input("Enter choice (1 or 2): ").strip()
    use_advanced = method == "1"
    
    while True:
        input_text = input("\n📝 Enter a news sentence to fact-check (or 'quit' to exit):\n> ")
        
        if input_text.lower() in ['quit', 'exit', 'q']:
            break
            
        claims = extract_claim(input_text)
        
        for claim in claims:
            print(f"\n🔍 Claim: {claim}")
            similar_facts = search_similar_facts(claim)
            
            print("📚 Similar Facts:")
            for i, (fact, similarity) in enumerate(similar_facts, 1):
                print(f" {i}. {fact} (similarity: {similarity:.3f})")
            
            if use_advanced:
                verdict = get_verdict_improved(claim, similar_facts)
            else:
                verdict = get_verdict_simple(claim, similar_facts)
                
            print("\n🧠 Verdict:")
            print(verdict)
            print("-" * 60)