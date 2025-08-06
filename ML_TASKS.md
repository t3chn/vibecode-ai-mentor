# ML Engineer Tasks - VibeCode AI Mentor

## 🎯 Role Overview
You are the ML engineer responsible for embeddings generation, vector search optimization, LLM integration, and ensuring high-quality code recommendations for the VibeCode AI Mentor hackathon project.

## 📋 Primary Responsibilities

### 1. Embeddings Generation
- Integrate Google Gemini API for code embeddings
- Implement OpenAI API as fallback option
- Optimize chunk size for embedding quality (512-2048 tokens)
- Create batch processing with retry logic
- Monitor API usage and costs

### 2. Vector Search Optimization
- Design vector similarity queries for TiDB
- Implement hybrid search (vector + metadata)
- Tune similarity thresholds
- Create relevance ranking algorithms
- Optimize search performance

### 3. LLM Integration
- Setup Gemini 1.5 Flash for recommendations
- Design effective prompt templates
- Implement few-shot learning examples
- Add response streaming for better UX
- Create fallback strategies

### 4. Pattern Recognition
- Build pattern matching system
- Create anti-pattern detection
- Design confidence scoring
- Implement semantic code similarity
- Develop quality metrics

## 🛠 Technical Requirements

### APIs and Models
```python
# Primary LLM
Model: gemini-1.5-flash
Context: 8192 tokens
Cost: $0.0001 per 1K tokens

# Embedding Model
Model: text-embedding-3-small
Dimensions: 1536
Cost: $0.02 per 1M tokens

# Fallback Options
- OpenAI GPT-3.5-turbo
- Local embeddings (sentence-transformers)
```

### Dependencies to Use
```python
# AI/ML
google-generativeai
openai >= 1.0
tiktoken
numpy
scikit-learn

# Utilities
tenacity  # Retry logic
asyncio   # Async operations
aiohttp   # Async HTTP
```

## 📁 Files You Own

```
src/
├── embeddings/
│   ├── __init__.py
│   ├── base.py          # Abstract embedding class
│   ├── gemini.py        # Gemini implementation
│   ├── openai.py        # OpenAI implementation
│   └── batch.py         # Batch processing
├── search/
│   ├── __init__.py
│   ├── vector_search.py # Vector similarity search
│   ├── hybrid_search.py # Combined search logic
│   └── ranker.py        # Result ranking
├── generator/
│   ├── __init__.py
│   ├── llm_client.py    # LLM abstraction
│   ├── prompts.py       # Prompt templates
│   └── formatter.py     # Response formatting
└── ml/
    ├── __init__.py
    ├── patterns.py      # Pattern recognition
    ├── metrics.py       # Quality metrics
    └── optimizer.py     # Search optimization
```

## 🔧 Key Algorithms

### 1. Smart Chunking Strategy
```python
# Optimal chunk sizes
MIN_CHUNK_SIZE = 512 tokens
MAX_CHUNK_SIZE = 2048 tokens
OVERLAP = 128 tokens

# Chunk by:
- Complete functions
- Logical code blocks
- Natural boundaries
```

### 2. Similarity Search
```sql
-- Hybrid query example
SELECT 
    cs.*,
    VEC_COSINE_DISTANCE(cs.embedding, ?) as similarity,
    bp.pattern_name
FROM code_snippets cs
JOIN best_practices bp ON ...
WHERE similarity < 0.3
ORDER BY similarity ASC
LIMIT 10;
```

### 3. Prompt Engineering
```python
SYSTEM_PROMPT = """You are an expert code reviewer.
Analyze the code and provide specific, actionable improvements.
Focus on: readability, performance, security, best practices."""

FEW_SHOT_EXAMPLES = [...]
```

## 🔄 Integration Points

### With Backend Developer
- Receive code chunks for embedding
- Return embedding vectors
- Provide search results
- Send generated recommendations

### With DevOps Engineer
- Monitor API costs
- Track performance metrics
- Optimize resource usage

### With QA Tester
- Validate recommendation quality
- Test edge cases
- Improve accuracy metrics

## 📊 Success Metrics

### Quality Targets
- Embedding generation < 1s per chunk
- Search accuracy > 85%
- Recommendation relevance > 4/5
- API costs < $10 for demo

### Performance Standards
- Batch processing 100 chunks/minute
- Vector search < 200ms
- LLM response < 3s
- Cache hit rate > 60%

## 🚀 Quick Start Commands

```bash
# Test embedding generation
python -m src.embeddings.gemini test

# Run similarity search
python -m src.search.vector_search query "def calculate_complexity"

# Generate recommendations
python -m src.generator.llm_client analyze ./sample.py

# Benchmark performance
python scripts/benchmark_ml.py
```

## ⏰ Timeline

### Phase 1 (Hours 6-14)
- [ ] Setup Gemini API integration
- [ ] Implement embedding generation
- [ ] Create chunking strategy
- [ ] Test with sample code

### Phase 2 (Hours 26-34)
- [ ] Design prompt templates
- [ ] Implement vector search
- [ ] Create ranking algorithm
- [ ] Add pattern matching

### Phase 3 (Hours 34-42)
- [ ] Optimize search quality
- [ ] Fine-tune prompts
- [ ] Add caching layer
- [ ] Performance optimization

## 🎯 Definition of Done

- [ ] Embeddings generated for 10K+ code snippets
- [ ] Vector search returning relevant results
- [ ] LLM generating quality recommendations
- [ ] All APIs properly rate-limited
- [ ] Costs tracked and optimized
- [ ] Demo-ready with impressive results

## 📚 Reference Patterns to Index

### High-Quality Examples
- Django ORM patterns
- FastAPI async handlers  
- SQLAlchemy models
- Pytest fixtures

### Common Anti-patterns
- God functions (>100 lines)
- Deeply nested code
- Mutable default arguments
- Bare except clauses