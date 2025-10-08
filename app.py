from flask import Flask, render_template_string, request, jsonify
import threading
import markdown
import pandas as pd
from pinecone_rag_setup import PropertyRAGAgent, get_pinecone_index, get_hf_model, get_gemini_model

app = Flask(__name__)

# Singletons (ensure instant reuse)
PINECONE_INDEX = get_pinecone_index()
HF_MODEL = get_hf_model()
GEMINI_MODEL = get_gemini_model()
RAG_AGENT = PropertyRAGAgent(pinecone_index=PINECONE_INDEX, gemini_model=GEMINI_MODEL, hf_model=HF_MODEL)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>🏠 Property AI Agent — Ultra Fast RAG</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0e1a; --card: #1a1f2e; --accent: #6366f1; --accent-light: #818cf8; 
      --muted: #94a3b8; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
      --text: #f1f5f9; --text-muted: #cbd5e1; --border: rgba(255,255,255,0.1);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: 'Inter', sans-serif; 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
      color: var(--text); 
      min-height: 100vh;
      overflow-x: hidden;
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    
    /* Header */
    .header { 
      display: flex; align-items: center; justify-content: space-between; 
      margin-bottom: 30px; padding: 20px 0;
    }
    .brand { 
      display: flex; align-items: center; gap: 12px;
      font-size: 28px; font-weight: 800; letter-spacing: -0.5px;
    }
    .brand .icon { color: var(--accent); font-size: 32px; }
    .brand .text { background: linear-gradient(135deg, var(--accent), var(--accent-light)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .status-badge { 
      display: flex; align-items: center; gap: 8px; 
      padding: 8px 16px; background: var(--card); border-radius: 20px; 
      border: 1px solid var(--border); font-size: 14px; font-weight: 500;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    
    /* Main Card */
    .main-card { 
      background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      backdrop-filter: blur(20px); border: 1px solid var(--border); 
      border-radius: 24px; padding: 32px; box-shadow: 0 25px 50px rgba(0,0,0,0.3);
      margin-bottom: 24px;
    }
    
    /* Search Section */
    .search-section { margin-bottom: 24px; }
    .search-box { 
      display: flex; gap: 16px; align-items: stretch; margin-bottom: 16px;
    }
    .search-input { 
      flex: 1; padding: 16px 20px; border-radius: 16px; border: 2px solid var(--border);
      background: rgba(15, 23, 42, 0.8); color: var(--text); font-size: 16px;
      outline: none; transition: all 0.3s ease; backdrop-filter: blur(10px);
    }
    .search-input:focus { 
      border-color: var(--accent); box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
      background: rgba(15, 23, 42, 0.9);
    }
    .search-input::placeholder { color: var(--muted); }
    .ask-btn { 
      padding: 16px 24px; border-radius: 16px; border: none; 
      background: linear-gradient(135deg, var(--accent), var(--accent-light));
      color: white; font-weight: 600; font-size: 16px; cursor: pointer;
      box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3); transition: all 0.3s ease;
      min-width: 120px;
    }
    .ask-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 35px rgba(99, 102, 241, 0.4); }
    .ask-btn:active { transform: translateY(0); }
    .ask-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
    
    /* Test Questions */
    .test-questions { margin-bottom: 20px; }
    .test-questions h4 { 
      color: var(--text-muted); font-size: 14px; font-weight: 600; 
      margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .question-chips { 
      display: flex; flex-wrap: wrap; gap: 8px; 
    }
    .question-chip { 
      padding: 8px 16px; background: var(--card); border: 1px solid var(--border);
      border-radius: 20px; font-size: 13px; color: var(--text-muted); cursor: pointer;
      transition: all 0.3s ease; user-select: none;
    }
    .question-chip:hover { 
      background: var(--accent); color: white; border-color: var(--accent);
      transform: translateY(-1px);
    }
    
    /* Result Section */
    .result { margin-top: 24px; }
    .answer { 
      background: var(--card); border-radius: 16px; padding: 24px; 
      border: 1px solid var(--border); margin-bottom: 20px;
      min-height: 100px;
    }
    .answer.loading { 
      display: flex; align-items: center; justify-content: center; 
      color: var(--muted); font-style: italic;
    }
    .answer.loading::before { 
      content: "🤖"; margin-right: 8px; animation: spin 1s linear infinite;
    }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    
    /* Toggle Panels */
    .toggle-panel { 
      background: var(--card); border-radius: 16px; padding: 20px; 
      border: 1px solid var(--border);
    }
    .toggle-buttons { 
      display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
    }
    .toggle-btn { 
      display: flex; align-items: center; gap: 8px; padding: 10px 16px;
      background: transparent; border: 1px solid var(--border); border-radius: 12px;
      color: var(--text-muted); cursor: pointer; font-size: 14px; font-weight: 500;
      transition: all 0.3s ease; user-select: none;
    }
    .toggle-btn:hover { 
      background: var(--accent); color: white; border-color: var(--accent);
      transform: translateY(-1px);
    }
    .toggle-btn.active { 
      background: var(--accent); color: white; border-color: var(--accent);
    }
    .toggle-btn i { font-size: 16px; }
    
    .toggle-content { 
      display: none; margin-top: 16px;
    }
    .toggle-content.active { display: block; }
    .code-block { 
      background: #0f172a; border: 1px solid var(--border); border-radius: 12px;
      padding: 16px; font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
      font-size: 13px; line-height: 1.5; overflow-x: auto; color: #e2e8f0;
    }
    .json-block { 
      background: #0f172a; border: 1px solid var(--border); border-radius: 12px;
      padding: 16px; font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
      font-size: 13px; line-height: 1.5; overflow-x: auto; color: #e2e8f0;
      max-height: 300px; overflow-y: auto;
    }
    
    /* Stats Grid */
    .stats-grid { 
      display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
      gap: 20px; margin-top: 24px;
    }
    .stat-card { 
      background: var(--card); border: 1px solid var(--border); border-radius: 16px;
      padding: 24px; text-align: center;
    }
    .stat-card h3 { 
      font-size: 18px; font-weight: 700; margin-bottom: 12px; 
      background: linear-gradient(135deg, var(--accent), var(--accent-light));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-card p { color: var(--text-muted); line-height: 1.6; margin-bottom: 16px; }
    .stat-badges { 
      display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
    }
    .stat-badge { 
      padding: 6px 12px; background: var(--accent); color: white; 
      border-radius: 20px; font-size: 12px; font-weight: 600;
    }
    .stat-badge.success { background: var(--success); }
    .stat-badge.warning { background: var(--warning); }
    
    /* Responsive */
    @media (max-width: 768px) {
      .container { padding: 16px; }
      .header { flex-direction: column; gap: 16px; text-align: center; }
      .search-box { flex-direction: column; }
      .ask-btn { width: 100%; }
      .toggle-buttons { flex-direction: column; }
      .stats-grid { grid-template-columns: 1fr; }
    }
    
    /* Markdown Styling */
    .markdown h1, .markdown h2, .markdown h3 { 
      color: var(--text); margin: 16px 0 8px 0; font-weight: 700;
    }
    .markdown p { margin: 8px 0; line-height: 1.6; }
    .markdown ul, .markdown ol { margin: 8px 0; padding-left: 24px; }
    .markdown li { margin: 4px 0; }
    .markdown code { 
      background: var(--card); padding: 2px 6px; border-radius: 4px; 
      font-family: 'Fira Code', monospace; font-size: 13px;
    }
    .markdown pre { 
      background: var(--card); padding: 16px; border-radius: 8px; 
      overflow-x: auto; margin: 16px 0;
    }
    .markdown blockquote { 
      border-left: 4px solid var(--accent); padding-left: 16px; 
      margin: 16px 0; color: var(--text-muted); font-style: italic;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <div class="brand">
        <i class="fas fa-home icon"></i>
        <span class="text">Property AI Agent</span>
      </div>
      <div class="status-badge">
        <div class="status-dot"></div>
        <span>All Systems Operational</span>
      </div>
    </div>

    <!-- Main Card -->
    <div class="main-card">
      <!-- Search Section -->
      <div class="search-section">
        <div class="search-box">
          <input id="q" type="text" class="search-input" 
                 placeholder="Ask anything about properties... e.g. 'Show me 2 bedroom apartments under £2000 in London'" />
          <button id="ask" class="ask-btn">
            <i class="fas fa-search"></i> Ask AI
          </button>
        </div>
        
        <!-- Test Questions -->
        <div class="test-questions">
          <h4>Try these questions:</h4>
          <div class="question-chips">
            <div class="question-chip" data-query="highest crime rate in which city?">🏙️ Highest crime city</div>
            <div class="question-chip" data-query="2 bedroom apartments under £1500">🏠 2BR under £1500</div>
            <div class="question-chip" data-query="new homes with 3+ bedrooms">🆕 New 3+ BR homes</div>
            <div class="question-chip" data-query="studio apartments in low crime areas">🏢 Safe studios</div>
            <div class="question-chip" data-query="compare prices between studio and 2 bedroom">📊 Studio vs 2BR</div>
            <div class="question-chip" data-query="properties with flood risk">🌊 Flood risk properties</div>
            <div class="question-chip" data-query="best value property type">💰 Best value types</div>
            <div class="question-chip" data-query="most expensive location">💎 Most expensive area</div>
          </div>
        </div>
      </div>

      <!-- Result Section -->
      <div class="result" id="result">
        <div id="answer" class="answer">
          <div style="text-align: center; color: var(--muted); padding: 40px;">
            <i class="fas fa-robot" style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;"></i>
            <p>Ask me anything about properties! I can help you find the perfect home using AI-powered analytics.</p>
          </div>
        </div>
        
        <!-- Toggle Panel -->
        <div class="toggle-panel">
          <div class="toggle-buttons">
            <div class="toggle-btn" data-target="#pandas-code">
              <i class="fas fa-code"></i>
              <span>Pandas Code</span>
            </div>
            <div class="toggle-btn" data-target="#pandas-output">
              <i class="fas fa-table"></i>
              <span>Analytics Data</span>
            </div>
            <div class="toggle-btn" data-target="#vector-matches">
              <i class="fas fa-search"></i>
              <span>Vector Matches</span>
            </div>
          </div>
          
          <div id="pandas-code" class="toggle-content">
            <h4 style="color: var(--text-muted); margin-bottom: 12px; font-size: 14px;">🤖 AI-Generated Pandas Code:</h4>
            <div class="code-block" id="pandas-code-content">Click "Pandas Code" to see the AI-generated pandas query...</div>
          </div>
          
          <div id="pandas-output" class="toggle-content">
            <h4 style="color: var(--text-muted); margin-bottom: 12px; font-size: 14px;">📊 Analytics Results:</h4>
            <div class="json-block" id="pandas-output-content">Click "Analytics Data" to see the pandas output...</div>
          </div>
          
          <div id="vector-matches" class="toggle-content">
            <h4 style="color: var(--text-muted); margin-bottom: 12px; font-size: 14px;">🔍 Vector Database Matches:</h4>
            <div class="json-block" id="vector-matches-content">Click "Vector Matches" to see semantic search results...</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="stats-grid">
      <div class="stat-card">
        <h3><i class="fas fa-brain"></i> AI Agent Status</h3>
        <p>Hybrid RAG system with semantic retrieval + dynamic pandas analytics. All AI models are cached for instant responses.</p>
        <div class="stat-badges">
          <div class="stat-badge success">Pinecone ✓</div>
          <div class="stat-badge success">HuggingFace ✓</div>
          <div class="stat-badge success">Gemini ✓</div>
        </div>
      </div>
      
      <div class="stat-card">
        <h3><i class="fas fa-database"></i> Data Pipeline</h3>
        <p>Real-time analytics with 147K+ property records. Dynamic pandas code generation and vector similarity search.</p>
        <div class="stat-badges">
          <div class="stat-badge">147K Properties</div>
          <div class="stat-badge">Real-time Analytics</div>
          <div class="stat-badge">Vector Search</div>
        </div>
      </div>
      
      <div class="stat-card">
        <h3><i class="fas fa-magic"></i> Features</h3>
        <p>Advanced AI capabilities including natural language queries, constraint extraction, and intelligent response generation.</p>
        <div class="stat-badges">
          <div class="stat-badge">NL Queries</div>
          <div class="stat-badge">Smart Filters</div>
          <div class="stat-badge">Markdown Output</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Markdown rendering
    const md = (txt) => window.marked ? marked.parse(txt || '') : (txt || '');
    
    // Toggle functionality
    document.addEventListener('click', (e) => {
      const toggle = e.target.closest('.toggle-btn');
      if (!toggle) return;
      
      const target = toggle.getAttribute('data-target');
      const content = document.querySelector(target);
      const allToggles = document.querySelectorAll('.toggle-btn');
      const allContents = document.querySelectorAll('.toggle-content');
      
      // Toggle active state
      allToggles.forEach(t => t.classList.remove('active'));
      allContents.forEach(c => c.classList.remove('active'));
      
      toggle.classList.add('active');
      if (content) content.classList.add('active');
    });
    
    // Test question chips
    document.addEventListener('click', (e) => {
      const chip = e.target.closest('.question-chip');
      if (!chip) return;
      
      const query = chip.getAttribute('data-query');
      if (query) {
        document.getElementById('q').value = query;
        ask();
      }
    });
    
    // Ask function
    async function ask() {
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      
      const btn = document.getElementById('ask');
      const answerDiv = document.getElementById('answer');
      
      // Show loading state
      btn.disabled = true;
      btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Thinking...';
      answerDiv.className = 'answer loading';
      answerDiv.innerHTML = 'AI Agent is analyzing your query...';
      
      try {
        const startTime = Date.now();
        const r = await fetch('/api/ask', { 
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' }, 
          body: JSON.stringify({ q }) 
        });
        
        const j = await r.json();
        const endTime = Date.now();
        const responseTime = ((endTime - startTime) / 1000).toFixed(2);
        
        if (j.error) {
          answerDiv.innerHTML = `<div style="color: var(--danger); padding: 20px; text-align: center;">
            <i class="fas fa-exclamation-triangle"></i><br>Error: ${j.error}
          </div>`;
        } else {
          answerDiv.className = 'answer';
          answerDiv.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);">
              <h3 style="margin: 0; color: var(--text);"><i class="fas fa-robot"></i> AI Response</h3>
              <span style="color: var(--muted); font-size: 12px;">${responseTime}s response time</span>
            </div>
            <div class="markdown">${md(j.answer)}</div>
          `;
          
          // Update toggle content
          document.getElementById('pandas-code-content').textContent = j.analytics.generated_code || 'No pandas code generated';
          document.getElementById('pandas-output-content').textContent = JSON.stringify(j.analytics.rows || [], null, 2);
          document.getElementById('vector-matches-content').textContent = JSON.stringify(j.matches || [], null, 2);
        }
      } catch (err) {
        answerDiv.className = 'answer';
        answerDiv.innerHTML = `<div style="color: var(--danger); padding: 20px; text-align: center;">
          <i class="fas fa-exclamation-triangle"></i><br>Error: ${err?.message || String(err)}
        </div>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-search"></i> Ask AI';
      }
    }
    
    // Event listeners
    document.getElementById('ask').addEventListener('click', ask);
    document.getElementById('q').addEventListener('keydown', (e) => { 
      if (e.key === 'Enter') ask(); 
    });
  </script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json(force=True)
    q = (data.get('q') or '').strip()
    if not q:
        return jsonify({"error":"empty query"}), 400
    
    try:
        # Run analytics first to expose generated code, then vector search
        print(f"[DEBUG] Processing query: {q}")
        analytics = RAG_AGENT.analytics_agent.analyze(q, top_k=7)
        print(f"[DEBUG] Analytics completed")
        matches = RAG_AGENT.search_properties(q, top_k=7)
        print(f"[DEBUG] Vector search completed: {len(matches)} matches")
        answer = RAG_AGENT.generate_response(q, matches)
        print(f"[DEBUG] Response generated")
        
        # Convert to JSON-serializable format
        analytics_serializable = {
            "context": str(analytics.get('context', '')),
            "generated_code": str(analytics.get('generated_code', '')),
            "rows": analytics.get('rows', [])
        }
        
        matches_serializable = []
        for match in matches:
            # Safely convert metadata values
            safe_metadata = {}
            for k, v in match.get('metadata', {}).items():
                try:
                    if pd.isna(v) or v is None:
                        safe_metadata[k] = ''
                    elif isinstance(v, (int, float, str, bool)):
                        safe_metadata[k] = str(v)
                    else:
                        safe_metadata[k] = str(v)
                except Exception:
                    safe_metadata[k] = ''
            
            matches_serializable.append({
                "id": str(match.get('id', '')),
                "score": float(match.get('score', 0)),
                "metadata": safe_metadata
            })
        
        return jsonify({
            "answer": str(answer) if answer else "No response generated",
            "analytics": analytics_serializable,
            "matches": matches_serializable
        })
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run threaded for responsiveness
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)

