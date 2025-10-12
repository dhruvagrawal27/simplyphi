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
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime

# Load .env
load_dotenv(override=True)

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Configuration
PINECONE_INDEX_NAME = "property-data-rag"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 embedding dimension
BATCH_SIZE = 250  # Process embeddings in batches

# Cached singletons
_CACHED_PINECONE_INDEX = None
_CACHED_HF_MODEL = None
_CACHED_GEMINI_MODEL = None
_CACHED_OPENAI_CLIENT = None

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

def load_analytics_data() -> pd.DataFrame:
    """Load cleaned dataset for analytics agent"""
    try:
        # Prefer cleaned dataset if available
        if os.path.exists('property_data_cleaned.csv'):
            df = pd.read_csv('property_data_cleaned.csv', encoding='utf-8')
        else:
            # Fallback to embeddings-ready
            df = pd.read_csv('property_embeddings_ready.csv', encoding='utf-8')
        # Normalize/parse dates if available
        if 'listing_update_date' in df.columns:
            df['listing_update_date'] = pd.to_datetime(df['listing_update_date'], errors='coerce')
            # Handle NaT values
            df['listing_update_date'] = df['listing_update_date'].fillna(pd.Timestamp('1900-01-01'))
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load analytics data: {e}")
        return pd.DataFrame()

class DataAnalyticsAgent:
    """Rule-based analytics agent that converts natural language to pandas filters and returns top rows context."""

    def __init__(self, analytics_df: pd.DataFrame):
        self.df = analytics_df.copy()
        self.schema = {col: str(self.df[col].dtype) for col in self.df.columns}

    def _compute_days_since_update_inline(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'listing_update_date' in df.columns:
            now = pd.Timestamp.now(tz=None)
            df['listing_update_date'] = pd.to_datetime(df['listing_update_date'], errors='coerce')
            # Handle NaT values properly
            df['days_since_update'] = (now - df['listing_update_date']).dt.days
            df['days_since_update'] = df['days_since_update'].fillna(999)  # Fill NaT with large number
        return df

    def analyze(self, nl_query: str, top_k: int = 7) -> Dict[str, Any]:
        """Use Gemini to produce a pandas query plan and execute it safely."""
        if self.df.empty:
            return {"context": "No analytics data available.", "rows": []}
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')

        schema_preview = "\n".join([f"- {c}: {t}" for c, t in self.schema.items()][:40])
        prompt = f"""
You are a senior pandas engineer. Convert the user's question into CONCISE pandas code over an existing DataFrame named df.

REQUIREMENTS:
- Return ONLY the essential pandas code (2-3 lines max)
- Use df, pandas (pd), numpy (np) only
- Check column existence: if 'col' in df.columns
- Build DataFrame 'result' with max {top_k} rows
- NO comments, NO explanations, NO test data

FILTERING RULES:
- Location: df['address'].str.contains('location', case=False, na=False)
- Counts: Use == for exact match, >= for "or more"
- Price: < for "under", > for "over", <= for "up to"
- Sort: ascending=True for "cheapest", descending=True for "expensive"
- Crime queries: Use df.groupby('address')['crime_score_weight'].mean() for city-level analysis

Available columns: {', '.join(list(self.schema.keys())[:15])}

Query: {nl_query}

Code:
        """

        try:
            gen = model.generate_content(prompt)
            code = gen.text or ""
        except Exception as e:
            code = ""

        # Extract code from markdown fences if present
        def extract_code(text: str) -> str:
            try:
                import re
                m = re.search(r"```[a-zA-Z0-9_\-]*\n([\s\S]*?)```", text)
                if m:
                    return m.group(1).strip()
                return text.strip()
            except Exception:
                return text

        code = extract_code(code)

        # Build a safe execution environment
        local_env: Dict[str, Any] = {}
        # Start from a working copy
        df = self.df.copy()
        df = self._compute_days_since_update_inline(df)
        local_env['pd'] = pd
        local_env['np'] = np
        local_env['df'] = df
        local_env['result'] = pd.DataFrame()

        # Safety: strip forbidden keywords and clean up code
        forbidden = ['os.', 'sys.', 'open(', 'subprocess', 'eval(', 'exec(', 'import ', '__', 'pickle', 'pathlib']
        safe_code = code
        
        # Remove forbidden keywords
        for bad in forbidden:
            safe_code = safe_code.replace(bad, '# removed ')
        
        # Clean up excessive comments and explanations
        lines = safe_code.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines, comments, and explanations
            if (line and 
                not line.startswith('#') and 
                not line.startswith('"""') and 
                not line.startswith("'''") and
                not line.startswith('try:') and
                not line.startswith('except') and
                not 'Initialize result' in line and
                not 'Define required' in line and
                not 'Check if all required' in line and
                not 'This part is for testing' in line):
                cleaned_lines.append(line)
        
        safe_code = '\n'.join(cleaned_lines)

        try:
            exec(safe_code, {}, local_env)
        except Exception:
            # fallback: try to apply basic filters based on query
            tmp = df.copy()
            cols = [c for c in ['type_standardized','property_type_full_description','bedrooms','bathrooms','price','address','is_new_home','listing_update_date','days_since_update','price_category'] if c in tmp.columns]
            
            # Apply basic filters based on query keywords
            query_lower = nl_query.lower()
            
            # Filter by bathrooms if mentioned
            if 'bathroom' in query_lower and 'bathrooms' in tmp.columns:
                if '2' in query_lower:
                    tmp = tmp[tmp['bathrooms'] == 2]
                elif '1' in query_lower:
                    tmp = tmp[tmp['bathrooms'] == 1]
                elif '3' in query_lower:
                    tmp = tmp[tmp['bathrooms'] == 3]
            
            # Filter by location if mentioned
            if 'london' in query_lower and 'address' in tmp.columns:
                tmp = tmp[tmp['address'].str.contains('london', case=False, na=False)]
            elif 'birmingham' in query_lower and 'address' in tmp.columns:
                tmp = tmp[tmp['address'].str.contains('birmingham', case=False, na=False)]
            elif 'manchester' in query_lower and 'address' in tmp.columns:
                tmp = tmp[tmp['address'].str.contains('manchester', case=False, na=False)]
            
            # Handle crime rate queries
            if 'crime' in query_lower and 'crime_score_weight' in tmp.columns and 'address' in tmp.columns:
                crime_by_city = tmp.groupby('address')['crime_score_weight'].mean().sort_values(ascending=False)
                if 'highest' in query_lower:
                    # Return top cities with highest crime + sample properties from those cities
                    result_data = []
                    top_cities = crime_by_city.head(3)  # Top 3 highest crime cities
                    
                    for city, crime_score in top_cities.items():
                        # Add city summary
                        result_data.append({
                            'address': city, 
                            'crime_score_weight': crime_score,
                            'property_type_full_description': f'City Summary',
                            'bedrooms': '',
                            'bathrooms': '',
                            'price': '',
                            'type_standardized': 'city_summary'
                        })
                        
                        # Add sample properties from this high-crime city
                        city_properties = tmp[tmp['address'] == city].head(2)
                        for _, prop in city_properties.iterrows():
                            result_data.append({
                                'address': city,
                                'crime_score_weight': crime_score,
                                'property_type_full_description': str(prop.get('property_type_full_description', '')),
                                'bedrooms': prop.get('bedrooms', ''),
                                'bathrooms': prop.get('bathrooms', ''),
                                'price': prop.get('price', ''),
                                'type_standardized': str(prop.get('type_standardized', ''))
                            })
                    
                    local_env['result'] = pd.DataFrame(result_data)
                else:
                    # Return top cities with lowest crime
                    result_data = []
                    for city, crime_score in crime_by_city.tail(top_k).items():
                        result_data.append({'address': city, 'crime_score_weight': crime_score})
                    local_env['result'] = pd.DataFrame(result_data)
            else:
                # Sort by price for cheapest queries
                if 'cheapest' in query_lower and 'price' in tmp.columns:
                    tmp = tmp.sort_values('price', ascending=True)
                elif 'expensive' in query_lower and 'price' in tmp.columns:
                    tmp = tmp.sort_values('price', ascending=False)
                elif 'days_since_update' in tmp.columns:
                    tmp = tmp.sort_values('days_since_update', ascending=True)
                elif 'price' in tmp.columns:
                    tmp = tmp.sort_values('price', ascending=True)
                
                local_env['result'] = tmp[cols].head(top_k)

        result_df = local_env.get('result')
        if not isinstance(result_df, pd.DataFrame):
            result_df = pd.DataFrame()
        # enforce top_k
        result_df = result_df.head(top_k)

        # Build concise context
        lines = []
        for _, row in result_df.iterrows():
            # Check if this is a crime analysis result
            if 'crime_score_weight' in result_df.columns and row.get('type_standardized') == 'city_summary':
                # Crime analysis format: City - Crime Score
                addr = row.get('address', '')
                crime_score = row.get('crime_score_weight', '')
                line = f"CRIME DATA: {addr} has the highest crime rate with score {crime_score}"
            else:
                # Property format: Property Type - £Price - Address (X bed, X bath) [New]
                desc = str(row.get('property_type_full_description', '')) or str(row.get('type_standardized', ''))
                bed = row.get('bedrooms', '')
                bath = row.get('bathrooms', '')
                price = row.get('price', '')
                addr = row.get('address', '')
                new_home = row.get('is_new_home', False)
                crime_score = row.get('crime_score_weight', '')
                
                line = f"{desc} - £{price} - {addr} ({bed} bed, {bath} bath)"
                if new_home:
                    line += " [New]"
                if crime_score and crime_score != '':
                    line += f" [Crime Score: {crime_score}]"
            lines.append(line)
        
        context = "\n".join(lines) if lines else "No matching data found."
        return {"context": context, "rows": result_df.to_dict(orient='records'), "generated_code": safe_code}

class ChatGPTAnalyticsAgent:
    """Analytics agent using OpenAI (gpt-4o-mini) to generate pandas code and execute safely."""

    def __init__(self, analytics_df: pd.DataFrame):
        self.df = analytics_df.copy()
        self.schema = {col: str(self.df[col].dtype) for col in self.df.columns}

    def _compute_days_since_update_inline(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'listing_update_date' in df.columns:
            now = pd.Timestamp.now(tz=None)
            df['listing_update_date'] = pd.to_datetime(df['listing_update_date'], errors='coerce')
            df['days_since_update'] = (now - df['listing_update_date']).dt.days
            df['days_since_update'] = df['days_since_update'].fillna(999)
        return df

    def analyze(self, nl_query: str, top_k: int = 7) -> Dict[str, Any]:
        if self.df.empty:
            return {"context": "No analytics data available.", "rows": [], "generated_code": ""}

        # Build prompt
        schema_preview = "\n".join([f"- {c}: {t}" for c, t in self.schema.items()][:40])
        prompt = f"""
You are a senior pandas engineer. Convert the user's question into CONCISE pandas code over an existing DataFrame named df.

REQUIREMENTS:
- Return ONLY the essential pandas code (2-3 lines max)
- Use df, pandas (pd), numpy (np) only
- Check column existence: if 'col' in df.columns
- Build DataFrame 'result' with max {top_k} rows
- NO comments, NO explanations, NO test data

FILTERING RULES:
- Location: df['address'].str.contains('location', case=False, na=False)
- Counts: Use == for exact match, >= for "or more"
- Price: < for "under", > for "over", <= for "up to"
- Sort: ascending=True for "cheapest", descending=True for "expensive"
- Crime queries: Use df.groupby('address')['crime_score_weight'].mean() for city-level analysis

Available columns: {', '.join(list(self.schema.keys())[:15])}

Query: {nl_query}

Code:
"""

        # Call OpenAI (primary)
        code = ""
        error_msg = ""
        try:
            client = get_openai_client()
            if client is None:
                error_msg = "OpenAI client not initialized (check OPENAI_API_KEY)"
                print(f"[WARN] {error_msg}")
            else:
                print(f"[DEBUG] Calling OpenAI API for query: {nl_query}")
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You only output python pandas code. No explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                code = (completion.choices[0].message.content or "").strip()
                print(f"[DEBUG] ChatGPT raw response: {code[:200]}")
        except Exception as e:
            error_msg = f"OpenAI API error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            code = ""

        # LangChain fallback if no code
        if not code:
            try:
                print("[DEBUG] Trying LangChain fallback...")
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
                response = llm.invoke(prompt)
                code = (response.content or "").strip()
                print(f"[DEBUG] LangChain response: {code[:200]}")
            except Exception as e:
                print(f"[ERROR] LangChain fallback failed: {str(e)}")
                code = ""

        # Extract code from markdown fences if present
        def extract_code(text: str) -> str:
            if not text:
                return ""
            try:
                import re
                # Try to find code in markdown fences
                m = re.search(r"```(?:python|py)?\n([\s\S]*?)```", text)
                if m:
                    extracted = m.group(1).strip()
                    print(f"[DEBUG] Extracted code from markdown: {extracted[:100]}")
                    return extracted
                # If no markdown, return as-is
                print(f"[DEBUG] No markdown fence found, using raw text")
                return text.strip()
            except Exception as e:
                print(f"[ERROR] Code extraction failed: {e}")
                return text.strip()

        code = extract_code(code)
        
        if not code:
            print(f"[WARN] No code generated for query: {nl_query}")
            print(f"[WARN] Error message: {error_msg}")
        else:
            print(f"[DEBUG] Final code to execute: {code}")

        # Safe execution environment
        local_env: Dict[str, Any] = {}
        df = self.df.copy()
        df = self._compute_days_since_update_inline(df)
        local_env['pd'] = pd
        local_env['np'] = np
        local_env['df'] = df
        local_env['result'] = pd.DataFrame()

        forbidden = ['os.', 'sys.', 'open(', 'subprocess', 'eval(', 'exec(', 'import ', '__', 'pickle', 'pathlib']
        safe_code = code
        for bad in forbidden:
            safe_code = safe_code.replace(bad, '# removed ')

        lines = safe_code.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if (line and 
                not line.startswith('#') and 
                not line.startswith('"""') and 
                not line.startswith("'''") ):
                cleaned_lines.append(line)
        safe_code = '\n'.join(cleaned_lines)

        try:
            print(f"[DEBUG] Executing safe code: {safe_code}")
            exec(safe_code, {}, local_env)
            print(f"[DEBUG] Code execution successful")
        except Exception as e:
            print(f"[ERROR] Code execution failed: {e}")
            # fallback heuristic mirroring DataAnalyticsAgent basics
            tmp = df.copy()
            cols = [c for c in ['type_standardized','property_type_full_description','bedrooms','bathrooms','price','address','is_new_home','listing_update_date','days_since_update','price_category'] if c in tmp.columns]
            query_lower = nl_query.lower()
            if 'london' in query_lower and 'address' in tmp.columns:
                tmp = tmp[tmp['address'].str.contains('london', case=False, na=False)]
            if 'cheapest' in query_lower and 'price' in tmp.columns:
                tmp = tmp.sort_values('price', ascending=True)
            elif 'expensive' in query_lower and 'price' in tmp.columns:
                tmp = tmp.sort_values('price', ascending=False)
            local_env['result'] = tmp[cols].head(top_k)

        result_df = local_env.get('result')
        if not isinstance(result_df, pd.DataFrame):
            result_df = pd.DataFrame()
        result_df = result_df.head(top_k)

        lines = []
        for _, row in result_df.iterrows():
            desc = str(row.get('property_type_full_description', '')) or str(row.get('type_standardized', ''))
            bed = row.get('bedrooms', '')
            bath = row.get('bathrooms', '')
            price = row.get('price', '')
            addr = row.get('address', '')
            new_home = row.get('is_new_home', False)
            line = f"{desc} - £{price} - {addr} ({bed} bed, {bath} bath)"
            if new_home:
                line += " [New]"
            lines.append(line)
        context = "\n".join(lines) if lines else "No matching data found."
        
        return {"context": context, "rows": result_df.to_dict(orient='records'), "generated_code": safe_code}


def setup_openai() -> Optional[object]:
    try:
        from openai import OpenAI
        if not OPENAI_API_KEY:
            print("[ERROR] OPENAI_API_KEY is not set in environment")
            print("[INFO] Please add OPENAI_API_KEY to your .env file")
            return None
        print("[INIT] Initializing OpenAI client...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Quick smoke test
        try:
            models = client.models.list()
            print(f"[OK] OpenAI client ready! (Found {len(list(models.data))} models)")
        except Exception as e:
            print(f"[WARN] OpenAI client created but test failed: {e}")
        return client
    except ImportError:
        print("[ERROR] openai package not installed. Installing...")
        os.system("pip install openai>=1.45.0")
        try:
            from openai import OpenAI  # noqa: F401
            return setup_openai()
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI after install: {e}")
            return None
    except Exception as e:
        print(f"[ERROR] Error setting up OpenAI: {e}")
        return None

def get_openai_client():
    global _CACHED_OPENAI_CLIENT
    if _CACHED_OPENAI_CLIENT is not None:
        return _CACHED_OPENAI_CLIENT
    client = setup_openai()
    # Only cache a successful client; don't cache None to allow later retries
    if client is not None:
        _CACHED_OPENAI_CLIENT = client
    return client

def setup_pinecone():
    """Initialize Pinecone client and create index"""
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        print("[INIT] Setting up Pinecone...")
        
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            print("[ERROR] Missing PINECONE_API_KEY")
            return None
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

def get_pinecone_index():
    global _CACHED_PINECONE_INDEX
    if _CACHED_PINECONE_INDEX is not None:
        return _CACHED_PINECONE_INDEX
    idx = setup_pinecone()
    _CACHED_PINECONE_INDEX = idx
    return idx

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

def get_hf_model():
    global _CACHED_HF_MODEL
    if _CACHED_HF_MODEL is not None:
        return _CACHED_HF_MODEL
    m = setup_hf_embeddings()
    _CACHED_HF_MODEL = m
    return m

def setup_gemini():
    """Initialize Gemini client for RAG responses"""
    try:
        import google.generativeai as genai
        
        print("[INIT] Setting up Gemini client...")
        if not GEMINI_API_KEY:
            print("[ERROR] Missing GEMINI_API_KEY")
            return None
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

def get_gemini_model():
    global _CACHED_GEMINI_MODEL
    if _CACHED_GEMINI_MODEL is not None:
        return _CACHED_GEMINI_MODEL
    g = setup_gemini()
    _CACHED_GEMINI_MODEL = g
    return g

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
    
    def __init__(self, pinecone_index=None, gemini_model=None, hf_model=None, analytics_df: pd.DataFrame=None):
        self.index = pinecone_index or get_pinecone_index()
        self.model = gemini_model or get_gemini_model()
        self.hf_model = hf_model or get_hf_model()
        analytics_df = load_analytics_data()
        self.analytics_agent = DataAnalyticsAgent(analytics_df)
        self.chatgpt_analytics_agent = ChatGPTAnalyticsAgent(analytics_df)
        self.vector_score_threshold = 0.6
        self.default_top_k = 15

    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        constraints: Dict[str, Any] = {}
        import re
        # bedrooms
        m = re.search(r'(\d+)\s*\+?\s*bed', q)
        if m:
            constraints['min_bedrooms'] = int(m.group(1))
        # price upper bound
        m = re.search(r'under\s*£?(\d+)', q) or re.search(r'<\s*£?(\d+)', q)
        if m:
            constraints['max_price'] = int(m.group(1))
        # location
        m = re.search(r'\bin\s+([a-z\s\-]+)', q)
        if m:
            constraints['location_like'] = m.group(1).strip()
        # new home
        if 'new home' in q or 'new homes' in q:
            constraints['is_new_home'] = True
        # crime constraints
        if 'highest crime' in q or 'high crime' in q:
            constraints['high_crime'] = True
        elif 'low crime' in q:
            constraints['low_crime'] = True
        return constraints

    def _metadata_matches(self, md: Dict[str, Any], cons: Dict[str, Any]) -> bool:
        # bedrooms
        if 'min_bedrooms' in cons:
            try:
                if int(md.get('bedrooms', 0)) < cons['min_bedrooms']:
                    return False
            except Exception:
                return False
        # price
        if 'max_price' in cons:
            try:
                if float(md.get('price', 1e12)) > cons['max_price']:
                    return False
            except Exception:
                return False
        # location
        if 'location_like' in cons:
            if cons['location_like'] and cons['location_like'].lower() not in str(md.get('address','')).lower():
                return False
        # new home
        if cons.get('is_new_home') is True and not bool(md.get('is_new_home', False)):
            return False
        # low crime
        if cons.get('low_crime') is True:
            try:
                if float(md.get('crime_score', 10.0)) > 3.0:
                    return False
            except Exception:
                return False
        # high crime
        if cons.get('high_crime') is True:
            try:
                if float(md.get('crime_score', 0.0)) < 8.0:  # Only show high crime areas (8+)
                    return False
            except Exception:
                return False
        return True
    
    def search_properties(self, query: str, top_k: int = 7) -> List[Dict]:
        """Search for relevant properties using semantic similarity"""
        try:
            constraints = self._extract_constraints(query)
            # Get embedding for the query
            query_embedding = get_embeddings_batch([query], self.hf_model)[0]
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=self.default_top_k,
                include_metadata=True
            )
            matches = results.get('matches', [])
            print(f"[DEBUG] Raw Pinecone matches: {len(matches)}")
            
            # Lower score threshold for better recall
            matches = [m for m in matches if m.get('score', 0) >= 0.3]  # Lowered from 0.6
            print(f"[DEBUG] After score filter: {len(matches)}")
            
            # post filter by constraints
            filtered = []
            for m in matches:
                md = m.get('metadata', {})
                if self._metadata_matches(md, constraints):
                    filtered.append(m)
            print(f"[DEBUG] After constraint filter: {len(filtered)}")
            
            # fallback if too few - return all matches regardless of constraints
            if len(filtered) < min(top_k, 3):
                print(f"[DEBUG] Using fallback - returning top matches")
                filtered = matches[:max(top_k, 3)]
            
            return filtered[:top_k]
         
        except Exception as e:
            print(f"[ERROR] Error searching properties: {e}")
            return []
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate response using Gemini based on retrieved properties"""
        try:
            # Backward-compat: if called without pre-analysis, run Gemini-only
            analytics_gemini = self.analytics_agent.analyze(query, top_k=7)
            analytics_chatgpt = self.chatgpt_analytics_agent.analyze(query, top_k=7)
            analytics_context = analytics_gemini.get('context', '')
            analytics_context_chatgpt = analytics_chatgpt.get('context', '')
            # sanitize function for currency artifacts
            def sanitize_text(s: str) -> str:
                return str(s).replace('[EMOJI]', '').replace('', '£')
            # Prepare context from search results
            context = "Property Information:\n"
            for i, result in enumerate(search_results, 1):
                metadata = result['metadata']
                context += f"{i}. {sanitize_text(metadata.get('text_description',''))}\n"
                context += f"   Similarity Score: {result['score']:.3f}\n\n"
            
            # Create prompt for Gemini
            prompt = f"""
You are Property RAG Assistant. Answer the user's question using the provided data.

User Query: {query}

Available Data:
Gemini Analytics:\n{analytics_context}\n\nChatGPT Analytics:\n{analytics_context_chatgpt}

Instructions:
- Answer the question directly using the provided data
- If the data shows crime scores, use them to answer crime-related questions
- If the data shows properties, list them with: type, bedrooms, bathrooms, price, address
- Use format: "Property Type - £X - Address (X bed, X bath)"
- Be confident and helpful - the data is accurate
- Don't say "data not available" if data is provided
"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"[ERROR] Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def analyze_both(self, query: str, top_k: int = 7) -> Dict[str, Any]:
        """Run both Gemini and ChatGPT analytics and return both results."""
        try:
            # Run sequentially here; callers may parallelize at higher level
            gem = self.analytics_agent.analyze(query, top_k=top_k)
            chg = self.chatgpt_analytics_agent.analyze(query, top_k=top_k)
            return {"gemini": gem, "chatgpt": chg}
        except Exception as e:
            print(f"[ERROR] analyze_both failed: {e}")
            return {"gemini": {"context": "", "rows": [], "generated_code": ""},
                    "chatgpt": {"context": "", "rows": [], "generated_code": ""}}

    def generate_response_with_context(self, query: str, search_results: List[Dict],
                                       analytics_gemini: Dict[str, Any], analytics_chatgpt: Dict[str, Any]) -> str:
        """Generate response using provided analytics contexts from both engines."""
        try:
            analytics_context = analytics_gemini.get('context', '') if analytics_gemini else ''
            analytics_context_chatgpt = analytics_chatgpt.get('context', '') if analytics_chatgpt else ''

            def sanitize_text(s: str) -> str:
                return str(s).replace('[EMOJI]', '').replace('', '£')

            context = "Property Information:\n"
            for i, result in enumerate(search_results, 1):
                metadata = result['metadata']
                context += f"{i}. {sanitize_text(metadata.get('text_description',''))}\n"
                context += f"   Similarity Score: {result['score']:.3f}\n\n"

            prompt = f"""
You are Property RAG Assistant. Answer the user's question using the provided data.

User Query: {query}

Available Data:
Gemini Analytics:\n{analytics_context}\n\nChatGPT Analytics:\n{analytics_context_chatgpt}

{context}

Instructions:
- Answer the question directly using the provided data
- If the data shows crime scores, use them to answer crime-related questions
- If the data shows properties, list them with: type, bedrooms, bathrooms, price, address
- Use format: "Property Type - £X - Address (X bed, X bath)"
- Be confident and helpful - the data is accurate
- Don't say "data not available" if data is provided
"""

            # Prefer OpenAI for final composition to avoid gRPC stalls
            try:
                client = get_openai_client()
            except Exception:
                client = None

            if client is not None:
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are Property RAG Assistant. Use given data to answer succinctly."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=700
                    )
                    return (completion.choices[0].message.content or "").strip()
                except Exception:
                    pass

            # Fallback to Gemini if OpenAI not available
            response = self.model.generate_content(prompt)
            return (getattr(response, 'text', None) or str(response))
        except Exception as e:
            print(f"[ERROR] Error generating response_with_context: {e}")
            return "I'm sorry, I encountered an error while processing your request."


def main():
    """Main function to set up the complete RAG system"""
    print("[INIT] Property Data RAG System Setup")
    print("=" * 50)
    
    # Step 1: Setup clients
    pinecone_index = get_pinecone_index()
    if not pinecone_index:
        return
    
    hf_model = get_hf_model()
    if not hf_model:
        return
    
    gemini_model = get_gemini_model()
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
    rag_agent = PropertyRAGAgent(pinecone_index=pinecone_index, gemini_model=gemini_model, hf_model=hf_model)
    
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

