"""
Quick Test Script for Property RAG System
=========================================

This script tests the RAG system with sample queries to ensure everything works correctly.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinecone_rag_setup import PropertyRAGAgent, setup_pinecone, setup_gemini, setup_hf_embeddings

def test_rag_system():
    """Test the RAG system with sample queries"""
    
    print("[TEST] Testing Property RAG System")
    print("=" * 40)
    
    try:
        # Setup clients
        print("[STEP 1] Setting up clients...")
        pinecone_index = setup_pinecone()
        hf_model = setup_hf_embeddings()
        gemini_model = setup_gemini()
        
        if not all([pinecone_index, hf_model, gemini_model]):
            print("[ERROR] Failed to setup clients")
            return False
        
        # Create RAG agent
        print("[STEP 2] Creating RAG agent...")
        rag_agent = PropertyRAGAgent(pinecone_index, gemini_model)
        
        # Test queries
        test_queries = [
            "Find me 2 bedroom apartments under £2000",
            "What are the cheapest properties?",
            "Show me properties with low crime scores",
            "Find new homes with 3+ bedrooms",
            "Properties in London under £1500"
        ]
        
        print("[STEP 3] Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n[QUERY] Test Query {i}: {query}")
            try:
                # Run analytics only to show context in test
                analytics = rag_agent.analytics_agent.analyze(query, top_k=7)
                print("[ANALYTICS CONTEXT]\n" + analytics.get('context',''))
                gen_code = analytics.get('generated_code','')
                if gen_code:
                    print("\n[GENERATED PANDAS CODE]\n" + gen_code)
                # Full RAG response
                response = rag_agent.chat(query)
                print(f"[OK] Response generated successfully")
                print(f"[PREVIEW] {response}...")
            except Exception as e:
                print(f"[ERROR] Error with query {i}: {e}")
                return False
        
        print("\n[OK] All tests passed! RAG system is working correctly.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_system()
    if success:
        print("\n[OK] RAG system is ready to use!")
    else:
        print("\n[FAIL] RAG system needs debugging.")

