from rag import retrieve, process_pdf

# Process the PDF first
process_pdf("Attention_Is_All_You_Need.pdf")

# Test retrieval directly
query = "What is the future work mentioned by the authors?"
chunks = retrieve(query, n_results=6)

print(f"Found {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk[:300])
    print()