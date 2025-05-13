import fitz  # PyMuPDF
from pathlib import Path

#________________Function to extract text from a PDF________________

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str or Path): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    pdf_path = Path(pdf_path)
    extracted_text = []

    #________________Safely open the PDF document________________
    
    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text.append(page.get_text())

    # ________________Join all page texts into a single string________________
    
    return "\n".join(extracted_text)

# ________________Main function________________

if __name__ == "__main__":
    
    # ________________Update your file path here________________
    
    pdf_file = r"D:\cusat\internship\Internship\week 2\may 13\Introduction to Large Language Model.pdf"

    try:
        # ________________Extract text from the specified PDF________________
        
        extracted_text = extract_text_from_pdf(pdf_file)
        
        # ________________Print the extracted text________________
        
        print(extracted_text)
    except Exception as e:
        print(f"An error occurred: {e}")
