## Build Log

### Day 1 — Core Pipeline
- Built PDF text extraction using PyMuPDF
- Implemented chunking with overlap
- Set up Chroma vector store
- Connected Groq LLaMA 3.3 for answer generation
- App runs locally — basic Q&A working

**Known issue:** Only 14 chunks generated from a 2.1MB paper — 
answers missing for some questions. Investigating chunking strategy.

### Day 2 — Chunking Fix
- Diagnosed low chunk count issue (14 chunks from 2.1MB paper)
- Root cause: chunk_size=500 too large for dense academic text
- Fix: reduced chunk_size to 200, overlap to 30
- Result: 60+ chunks created, retrieval accuracy improved significantly
- "explain attention visualizations" now returns correct answer

**Lesson learned:** Chunk size is not a set-and-forget parameter. 
Dense technical documents need smaller chunks than general text.