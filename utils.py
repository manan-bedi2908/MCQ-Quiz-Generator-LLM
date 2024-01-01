import PyPDF2
import json
import traceback

def parse_file(file):
    if file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
            return text
        except PyPDF2.utils.PdfReadError:
            raise Exception('Error reading the PDF File')
    
    elif file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    
    else:
        raise Exception(
            'Unsupported File Format. Only PDF and Text Files are supported'
        )