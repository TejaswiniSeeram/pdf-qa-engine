## Build Log

### Day 1 — Core Pipeline
- Built PDF text extraction using PyMuPDF
- Implemented chunking with overlap
- Set up Chroma vector store
- Connected Groq LLaMA 3.3 for answer generation
- App runs locally — basic Q&A working

**Known issue:** Only 14 chunks generated from a 2.1MB paper — 
answers missing for some questions. Investigating chunking strategy.
