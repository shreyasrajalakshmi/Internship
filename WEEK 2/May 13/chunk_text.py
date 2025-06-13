import fitz  # PyMuPDF

def extract_and_chunk_pdf(file_path, max_chunk_size=500, overlap=50):
    
    # Step 1: Extract text from PDF
    with fitz.open(file_path) as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()

    # Step 2: Chunk the extracted text
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + max_chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk.strip())
        start += max_chunk_size - overlap

    return chunks

file_path = "May 13\sample_ai_content.pdf"
chunks = extract_and_chunk_pdf(file_path)

# Print sample chunks
for i, chunk in enumerate(chunks[:3]):  # just first 3 chunks
    print(f"--- Chunk {i+1} ---\n{chunk}\n")


