import fitz  # PyMuPDF

def extract_pdf_text(filepath):
    """Extract text from a PDF document."""
    text = ""
    
    
    try:
        with fitz.open(filepath) as doc:
            print(f"Number of pages: {len(doc)}")
            for page_num, page in enumerate(doc, start=1):
                text += f"\n\n-- Page {page_num} --\n"
                text += page.get_text()
        print("PDF text extraction complete.")
        return text
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        

# Usage
pdf_text = extract_pdf_text('D:\cusat\internship\Internship\week 2\may 12\Introduction to Large Language Model.pdf')
if pdf_text:
    
   # ___________________ Display only the first 1000 characters___________________ 
    print(pdf_text[:1000]) 