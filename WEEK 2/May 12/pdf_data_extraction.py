import fitz  # PyMuPDF import

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc: #page wise text extraction
        for page in doc:
            text += page.get_text()
    return text


file_path = r'C:\Users\sahee\OneDrive\Desktop\Lealabs\WEEK 2\May 12\Bonafide Certificate Format.pdf'
pdf_text = extract_text_from_pdf(file_path)
print(pdf_text)
