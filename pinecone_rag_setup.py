"""
Property Data RAG System with Pinecone Vector Database
====================================================

This script creates a complete RAG (Retrieval-Augmented Generation) system for property data:
1. Loads property embeddings from CSV
2. Creates Pinecone index with Mistral embeddings
3. Stores vectors in Pinecone database
4. Creates RAG agent with Gemini for querying

Author: Assistant
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import time
from datetime import datetime

# API Keys (Store securely in production)
PINECONE_API_KEY = "pcsk_5YzRdG_Gbi5jtvXU1auarZuE5cAWkW4tTCYcVvcjzuTfG2PRxTrUgnkfhEUTBYE3DT3uYe"
GEMINI_API_KEY = "AIzaSyAwNwDXYoAo7m2SeaRUAmluHb5Z8e5IG5I"

# Configuration
PINECONE_INDEX_NAME = "property-data-rag"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 embedding dimension
BATCH_SIZE = 250  # Process embeddings in batches

def safe_int(value) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0

def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0

def setup_pinecone():
    """Initialize Pinecone client and create index"""
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        print("[INIT] Setting up Pinecone...")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index already exists
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME in existing_indexes:
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            # get dimension from describe_index (not from stats)
            try:
                desc = pc.describe_index(PINECONE_INDEX_NAME)
                current_dim = getattr(desc, 'dimension', None) or (desc.get('dimension') if isinstance(desc, dict) else None)
            except Exception:
                current_dim = None
            print(f"[INFO] Index '{PINECONE_INDEX_NAME}' already exists (dim={current_dim})")
            # If dimension mismatch, recreate index
            if current_dim and current_dim != EMBEDDING_DIMENSION:
                print(f"[WARNING] Dimension mismatch: {current_dim} != {EMBEDDING_DIMENSION}. Recreating index...")
                pc.delete_index(PINECONE_INDEX_NAME)
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                print("[WAIT] Waiting for index to be ready...")
                while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                    time.sleep(1)
                index = pc.Index(PINECONE_INDEX_NAME)
            else:
                print(f"[INFO] Index stats: {stats}")
            return index
        else:
            print(f"[CREATE] Creating new index '{PINECONE_INDEX_NAME}'...")
            
            # Create index with serverless configuration
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",  # Good for text embeddings
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            # Wait for index to be ready
            print("[WAIT] Waiting for index to be ready...")
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            
            print("[OK] Index created successfully!")
            index = pc.Index(PINECONE_INDEX_NAME)
            return index
            
    except ImportError:
        print("[ERROR] Pinecone package not installed. Installing...")
        os.system("pip install pinecone")
        return setup_pinecone()
    except Exception as e:
        print(f"[ERROR] Error setting up Pinecone: {e}")
        return None

def setup_hf_embeddings():
    """Initialize HuggingFace SentenceTransformer model for embeddings"""
    try:
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        os.environ["USE_TF"] = "0"
        os.environ["KERAS_BACKEND"] = "torch"
        from sentence_transformers import SentenceTransformer
        print("[INIT] Setting up HuggingFace SentenceTransformer model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # quick test
        _ = model.encode(["test"], normalize_embeddings=True)
        print("[OK] HuggingFace model ready! Embedding dimension: 384")
        return model
    except ImportError:
        print("[ERROR] sentence-transformers not installed. Installing...")
        os.system("pip install sentence-transformers")
        return setup_hf_embeddings()
    except Exception as e:
        if "Keras is Keras 3" in str(e) or "Keras 3" in str(e):
            print("[WARN] Keras 3 detected. Installing tf-keras for compatibility...")
            os.system("pip install tf-keras")
            try:
                os.environ["KERAS_BACKEND"] = "torch"
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                _ = model.encode(["test"], normalize_embeddings=True)
                print("[OK] HuggingFace model ready after tf-keras install! Embedding dimension: 384")
                return model
            except Exception as e2:
                print(f"[ERROR] Still failed to init HF embeddings: {e2}")
                return None
        print(f"[ERROR] Error setting up HF embeddings: {e}")
        return None

def setup_gemini():
    """Initialize Gemini client for RAG responses"""
    try:
        import google.generativeai as genai
        
        print("[INIT] Setting up Gemini client...")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Test the connection
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Hello, this is a test.")
        print("[OK] Gemini client ready!")
        return model
        
    except ImportError:
        print("[ERROR] Google Generative AI package not installed. Installing...")
        os.system("pip install google-generativeai")
        return setup_gemini()
    except Exception as e:
        print(f"[ERROR] Error setting up Gemini: {e}")
        return None

def get_embeddings_batch(texts: List[str], hf_model) -> List[List[float]]:
    """Get embeddings for a batch of texts using HuggingFace model"""
    try:
        vectors = hf_model.encode(texts, normalize_embeddings=True).tolist()
        return vectors
    except Exception as e:
        print(f"[ERROR] Error getting embeddings: {e}")
        return []

def process_property_data():
    """Load and process property embeddings data"""
    try:
        print("[LOAD] Loading property embeddings data...")
        
        # Load the embeddings-ready CSV
        df = pd.read_csv('property_embeddings_ready.csv', encoding='utf-8')
        print(f"[OK] Loaded {len(df)} properties")
        
        # Display sample data
        print("\n[INFO] Sample data:")
        print(df[['type_standardized', 'bedrooms', 'bathrooms', 'price', 'address']].head())
        
        return df
        
    except FileNotFoundError:
        print("[ERROR] property_embeddings_ready.csv not found!")
        print("Please run the EDA script first to generate the embeddings data.")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None

def upload_to_pinecone(df: pd.DataFrame, index, hf_model):
    """Upload property data to Pinecone with embeddings"""
    try:
        print(f"[UPLOAD] Uploading {len(df)} properties to Pinecone...")
        
        total_uploaded = 0
        
        # Process in batches
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i+BATCH_SIZE]
            
            print(f"[BATCH] Processing {i//BATCH_SIZE + 1}/{(len(df)-1)//BATCH_SIZE + 1} ({len(batch_df)} items)")
            
            # Get embeddings for this batch
            texts = batch_df['text_description'].tolist()
            embeddings = get_embeddings_batch(texts, hf_model)
            
            if not embeddings:
                print("[ERROR] Failed to get embeddings for batch")
                continue
            
            # Prepare vectors for Pinecone
            vectors = []
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                vector_id = f"property_{row.name}"  # Use DataFrame index as ID
                
                # Create metadata (store all relevant info)
                metadata = {
    'type': str(row['type_standardized']),
    'bedrooms': safe_int(row['bedrooms']),
    'bathrooms': safe_float(row['bathrooms']),
    'price': safe_int(row['price']),
    'address': str(row['address']),
    'crime_score': safe_float(row.get('crime_score_weight', 0)),
    'price_category': str(row.get('price_category', 'unknown')),
    'is_new_home': bool(row.get('is_new_home', False)),
    'flood_risk': bool(str(row.get('flood_risk', 'None')).lower() != 'none'),
    'text_description': str(row.get('text_description', ''))
                }

                
                vectors.append({
                    'id': vector_id,
                    'values': embeddings[idx],
                    'metadata': metadata
                })
            
            # Upload to Pinecone
            index.upsert(vectors=vectors)
            total_uploaded += len(vectors)
            
            print(f"[OK] Uploaded {len(vectors)} vectors (Total: {total_uploaded})")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        print(f"[OK] Successfully uploaded {total_uploaded} properties to Pinecone!")
        
        # Wait for indexing to complete
        print("[WAIT] Waiting for indexing to complete...")
        time.sleep(5)
        
        # Get final stats
        stats = index.describe_index_stats()
        print(f"[INFO] Final index stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error uploading to Pinecone: {e}")
        return False

class PropertyRAGAgent:
    """RAG Agent for property data queries"""
    
    def __init__(self, pinecone_index, gemini_model):
        self.index = pinecone_index
        self.model = gemini_model
        self.hf_model = setup_hf_embeddings()
        
    def search_properties(self, query: str, top_k: int = 7) -> List[Dict]:
        """Search for relevant properties using semantic similarity"""
        try:
            # Get embedding for the query
            query_embedding = get_embeddings_batch([query], self.hf_model)[0]
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results['matches']
            
        except Exception as e:
            print(f"[ERROR] Error searching properties: {e}")
            return []
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate response using Gemini based on retrieved properties"""
        try:
            # Prepare context from search results
            context = "Property Information:\n"
            for i, result in enumerate(search_results, 1):
                metadata = result['metadata']
                context += f"{i}. {metadata['text_description']}\n"
                context += f"   Similarity Score: {result['score']:.3f}\n\n"
            
            # Create prompt for Gemini
            prompt = f"""
            Based on the following property information, please answer the user's query in a helpful and informative way.
            
            User Query: {query}
            
            {context}
            
            Please provide a comprehensive answer based on the property data above. Include specific details like prices, locations, and property features when relevant.
            """
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"[ERROR] Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def chat(self, query: str) -> str:
        """Main chat function that combines search and generation"""
        print(f"\n[SEARCH] Query: {query}")
        
        # Search for relevant properties
        search_results = self.search_properties(query, top_k=5)
        
        if not search_results:
            return "I couldn't find any relevant properties for your query. Please try rephrasing your question."
        
        print(f"📋 Found {len(search_results)} relevant properties")
        
        # Generate response
        response = self.generate_response(query, search_results)
        
        # Also show the raw search results for transparency
        print("\n[INFO] Search Results:")
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            print(f"{i}. {metadata['type']} - £{metadata['price']} - {metadata['address']} (Score: {result['score']:.3f})")
        
        return response

def main():
    """Main function to set up the complete RAG system"""
    print("[INIT] Property Data RAG System Setup")
    print("=" * 50)
    
    # Step 1: Setup clients
    pinecone_index = setup_pinecone()
    if not pinecone_index:
        return
    
    hf_model = setup_hf_embeddings()
    if not hf_model:
        return
    
    gemini_model = setup_gemini()
    if not gemini_model:
        return
    
    # Step 2: Load property data
    df = process_property_data()
    if df is None:
        return
    
    # Step 3: Upload to Pinecone (only if index is empty)
    stats = pinecone_index.describe_index_stats()
    if stats['total_vector_count'] == 0:
        # Optional: limit upload size via env var for quick smoke test
        limit = os.environ.get('SAMPLE_UPLOAD_LIMIT')
        if limit and limit.isdigit():
            df = df.head(int(limit))
            print(f"[INFO] Using SAMPLE_UPLOAD_LIMIT={limit} rows for upload")
        print("[INFO] Index is empty, uploading data...")
        success = upload_to_pinecone(df, pinecone_index, hf_model)
        if not success:
            return
    else:
        print(f"[INFO] Index already has {stats['total_vector_count']} vectors")
    
    # Step 4: Create RAG agent
    print("\n[INIT] Creating RAG Agent...")
    rag_agent = PropertyRAGAgent(pinecone_index, gemini_model)
    
    print("[OK] RAG System is ready!")
    
    # Step 5: Optional interactive chat (skip when HEADLESS=1)
    if os.environ.get('HEADLESS') == '1':
        print('[INFO] HEADLESS mode enabled; skipping interactive chat.')
        return
    print("\n" + "=" * 50)
    print("[CHAT] Property RAG Chat - Type 'quit' to exit")
    print("=" * 50)
    while True:
        try:
            query = input("\n[INPUT] Ask about properties: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("[EXIT] Goodbye!")
                break
            if not query:
                continue
            response = rag_agent.chat(query)
            print(f"\n[RESPONSE]\n{response}")
        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] Error: {e}")

if __name__ == "__main__":
    main()

