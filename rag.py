import os
import fitz
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.EphemeralClient()
collection = None


# STEP 1: Extract
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    return pages


# STEP 2: Chunk — one chunk per page, then split large pages
def chunk_text(pages, max_words=500, overlap=50):
    chunks = []

    for page_text in pages:
        # Clean the text
        lines = page_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip references, page numbers, very short lines
            if len(line) < 20:
                continue
            if line.startswith('[') and line[1:3].strip().isdigit():
                continue
            cleaned_lines.append(line)

        page_clean = " ".join(cleaned_lines)
        words = page_clean.split()

        if not words:
            continue

        # If page is short enough — keep as one chunk
        if len(words) <= max_words:
            if len(words) > 20:
                chunks.append(page_clean)
        else:
            # Split large pages into smaller chunks with overlap
            for i in range(0, len(words), max_words - overlap):
                chunk = " ".join(words[i:i + max_words])
                if len(chunk.split()) > 20:
                    chunks.append(chunk)

    return chunks


# STEP 3: Embed
def embed(chunks):
    return embedder.encode(chunks).tolist()


# STEP 4: Store
def store(chunks, embeddings):
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )


# STEP 5: Retrieve
def retrieve(query, n_results=6):
    count = collection.count()
    n = min(n_results, count)
    if n == 0:
        return []
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n
    )
    return results["documents"][0]


# STEP 6: Answer
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant that answers questions 
based strictly on the document provided below.
If the answer is not clearly in the document, say: 
'I could not find this information in the document.'
Do not make up any information.

Document content:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1
    )
    return response.choices[0].message.content


# Main pipeline
def process_pdf(pdf_path):
    global collection

    try:
        chroma_client.delete_collection("pdf_qa")
    except Exception:
        pass
    collection = chroma_client.create_collection("pdf_qa")

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)
    embeddings = embed(chunks)
    store(chunks, embeddings)

    return len(chunks)


def answer_question(query):
    if collection is None:
        raise RuntimeError("No PDF processed yet.")

    context_chunks = retrieve(query)
    if not context_chunks:
        return "No relevant content found.", []

    answer = generate_answer(query, context_chunks)
    return answer, context_chunks